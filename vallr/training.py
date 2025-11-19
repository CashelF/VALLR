"""Training utilities and loops for VALLR models."""

from itertools import zip_longest
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import wandb
from torch.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import WarmupScheduler
from data.dataset import VideoDataset
# from Models.ML_VALLR import ML_VALLR
from Models.VALLR import VALLR
from transformers import VideoMAEConfig, Wav2Vec2Config


Tensor = torch.Tensor


def monitor_gradients(model: nn.Module) -> float:
    """Print and return the global gradient norm for debugging purposes."""
    total_norm = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.norm(2)
            total_norm += param_norm.item() ** 2
            print(f"{name} grad norm: {param_norm.item()}")
    total_norm = total_norm ** 0.5
    print(f"Total gradient norm: {total_norm}")
    return total_norm


def clamp_logits(logits: Tensor, min_logit: float = -10, max_logit: float = 10) -> Tensor:
    """Clamp logits to avoid numerical instabilities."""
    return torch.clamp(logits, min_logit, max_logit)


def custom_collate(batch: Iterable[Tuple[Tensor, Tensor]], phoneme_vocab: Dict[str, int]) -> Tuple[Tensor, List[Tensor]]:
    """Collate function that filters out missing samples and stacks valid tensors."""
    del phoneme_vocab  # Unused but kept for signature compatibility with original code.
    valid_batch = [item for item in batch if item is not None]

    if len(valid_batch) == 0:
        return torch.empty(0), torch.empty(0)

    videos, labels = zip(*valid_batch)
    videos = torch.stack(videos)
    label_tensors = [label.clone().detach() for label in labels]
    return videos, label_tensors


def log_all_metrics(epoch: int, epochs: int, train_loss: float, train_acc: float, val_loss: float, val_acc: float, lr: float) -> None:
    """Log metrics locally and to Weights & Biases."""
    print(
        f"Epoch [{epoch}/{epochs}], "
        f"Training Loss: {train_loss:.4f}, "
        f"Training Accuracy: {train_acc:.2f}%, "
        f"Validation Loss: {val_loss:.4f}, "
        f"Validation Accuracy: {val_acc:.2f}%, "
        f"Learning Rate: {lr:.8f}"
    )

    wandb.log(
        {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "learning_rate": lr,
        }
    )


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
    phoneme_vocab: Dict[str, int],
) -> Tuple[float, float]:
    """Train the model for a single epoch using mixed precision."""

    model.train()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0
    scaler = GradScaler()

    reverse_vocab = {v: k for k, v in phoneme_vocab.items()}

    for batch in tqdm(dataloader, desc="Training", leave=False):
        videos, labels = batch

        if videos.size(0) == 0:
            print("Skipping empty batch")
            continue

        labels = [label.to(device) for label in labels]
        videos = videos.to(device).float()

        optimizer.zero_grad()

        with autocast("cuda"):
            raw_logits, _ = model(videos)
            raw_logits = raw_logits.log_softmax(dim=-1)
            transpose_logits = raw_logits.transpose(0, 1)

            batch_size = transpose_logits.size(1)
            input_lengths = torch.full(size=(batch_size,), fill_value=transpose_logits.size(0), dtype=torch.long).to(device)
            target_lengths = torch.tensor([label.size(0) for label in labels], dtype=torch.long).to(device)

            if input_lengths.min() < target_lengths.max():
                print(
                    f"Skipping batch: input lengths {input_lengths.min()} < target lengths {target_lengths.max()}"
                )
                max_idx = torch.argmax(target_lengths).item()
                print(f"Longest target length index: {max_idx}")
                print(f"Target tensor (longest): {labels[max_idx].tolist()}")
                continue

            loss = criterion(transpose_logits, torch.cat(labels), input_lengths, target_lengths)

            if loss.item() < 0.0:
                print("Negative loss detected. Skipping...")
                continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        predicted_indices = torch.argmax(raw_logits, dim=-1)
        predicted_phonemes: List[List[str]] = []
        for batch_idx in range(predicted_indices.size(0)):
            frame_sequence = predicted_indices[batch_idx].tolist()
            decoded_phoneme_seq: List[str] = []
            previous_phoneme = None

            for timestep_idx in frame_sequence:
                phoneme = reverse_vocab.get(timestep_idx)
                if phoneme == "<pad>" or phoneme == previous_phoneme:
                    continue
                decoded_phoneme_seq.append(phoneme)
                previous_phoneme = phoneme

            predicted_phonemes.append(decoded_phoneme_seq)

        true_phonemes = []
        for label_seq in labels:
            phoneme_seq = [reverse_vocab[idx.item()] for idx in label_seq if idx.item() in reverse_vocab]
            true_phonemes.append(phoneme_seq)

        for pred_seq, true_seq in zip_longest(predicted_phonemes, true_phonemes, fillvalue=[]):
            correct_predictions = sum(
                p == t
                for p, t in zip_longest(pred_seq, true_seq, fillvalue=None)
                if p is not None and t is not None
            )
            total_correct += correct_predictions
            total_samples += len([t for t in true_seq if t is not None])

    avg_loss = running_loss / len(dataloader)
    accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0.0

    torch.cuda.empty_cache()

    return avg_loss, accuracy


def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    phoneme_vocab: Dict[str, int],
) -> Tuple[float, float]:
    """Evaluate the model for a single epoch using mixed precision."""

    model.eval()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0
    reverse_vocab = {v: k for k, v in phoneme_vocab.items()}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            videos, labels = batch

            if videos.size(0) == 0:
                print("Skipping empty batch")
                continue

            labels = [label.to(device) for label in labels]
            videos = videos.to(device).float()

            with autocast("cuda"):
                raw_logits, _ = model(videos)
                raw_logits = raw_logits.log_softmax(dim=-1)
                transpose_logits = raw_logits.transpose(0, 1)

                batch_size = transpose_logits.size(1)
                input_lengths = torch.full(size=(batch_size,), fill_value=transpose_logits.size(0), dtype=torch.long).to(device)
                target_lengths = torch.tensor([label.size(0) for label in labels], dtype=torch.long).to(device)

                if input_lengths.min() < target_lengths.max():
                    print(
                        f"Skipping batch: input lengths {input_lengths.min()} < target lengths {target_lengths.max()}"
                    )
                    continue

                loss = criterion(transpose_logits, torch.cat(labels), input_lengths, target_lengths)

                if loss.item() < 0.0:
                    print("Negative loss detected. Skipping...")
                    continue

            running_loss += loss.item()

            predicted_indices = torch.argmax(raw_logits, dim=-1)
            predicted_phonemes: List[List[str]] = []
            for batch_idx in range(predicted_indices.size(0)):
                frame_sequence = predicted_indices[batch_idx].tolist()
                decoded_phoneme_seq: List[str] = []
                previous_phoneme = None

                for timestep_idx in frame_sequence:
                    phoneme = reverse_vocab.get(timestep_idx)
                    if phoneme == "<pad>" or phoneme == previous_phoneme:
                        continue
                    decoded_phoneme_seq.append(phoneme)
                    previous_phoneme = phoneme

                predicted_phonemes.append(decoded_phoneme_seq)

            true_phonemes = []
            for label_seq in labels:
                phoneme_seq = [reverse_vocab[idx.item()] for idx in label_seq if idx.item() in reverse_vocab]
                true_phonemes.append(phoneme_seq)

            for pred_seq, true_seq in zip_longest(predicted_phonemes, true_phonemes, fillvalue=[]):
                correct_predictions = sum(
                    p == t
                    for p, t in zip_longest(pred_seq, true_seq, fillvalue=None)
                    if p is not None and t is not None
                )
                total_correct += correct_predictions
                total_samples += len([t for t in true_seq if t is not None])

    avg_loss = running_loss / len(dataloader)
    accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0.0

    torch.cuda.empty_cache()

    return avg_loss, accuracy


def save_model(model: nn.Module, save_model_path: str) -> None:
    """Persist the model checkpoint to disk."""
    import os

    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
    torch.save(model.state_dict(), save_model_path)
    print(f"Model saved to {save_model_path}")


def _build_model(version: str, phoneme_vocab: Dict[str, int]) -> nn.Module:
    """Instantiate a VALLR model based on the requested version."""
    if version == "V1":
        videomae_config = VideoMAEConfig()
        wav2vec_config = Wav2Vec2Config()
        wav2vec_config.vocab_size = len(phoneme_vocab)
        model = VALLR(
            videomae_config=videomae_config,
            wav2vec_config=wav2vec_config,
            adapter_dim=256,
        )
    # elif version == "V2":
    #     model = ML_VALLR(
    #         adapter_dim=256,
    #         num_classes=len(phoneme_vocab),
    #     )
    else:
        raise ValueError(f"Unsupported model version: {version}")

    return model


def train(
    device: torch.device,
    version: str,
    video_path: str,
    batch_size: int,
    num_workers: int,
    epochs: int,
    save_model_path: str,
    sample_size: float,
    vocab: Dict[str, int],
) -> None:
    """High-level training routine that mirrors the original main.py implementation."""

    torch.cuda.empty_cache()
    phoneme_vocab = vocab

    model = _build_model(version, phoneme_vocab)

    print(
        f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters."
    )

    criterion = nn.CTCLoss(blank=phoneme_vocab["<pad>"], reduction="mean", zero_infinity=True)
    lr = 1e-6
    target_lr = 1e-4
    warmup_steps = 500

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    after_scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    scheduler = WarmupScheduler(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        base_lr=lr,
        target_lr=target_lr,
        after_scheduler=after_scheduler,
    )

    training_dataset = VideoDataset(
        video_dir=video_path,
        split="train",
        num_frames=16,
        frame_size=(224, 224),
        phoneme_vocab=phoneme_vocab,
    )

    validation_dataset = VideoDataset(
        video_dir=video_path,
        split="val",
        num_frames=16,
        frame_size=(224, 224),
        phoneme_vocab=phoneme_vocab,
    )

    train_loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda batch: custom_collate(batch, phoneme_vocab),
    )

    val_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda batch: custom_collate(batch, phoneme_vocab),
    )

    subset_percentage = sample_size
    train_subset_size = int(len(training_dataset) * subset_percentage)
    val_subset_size = int(len(validation_dataset) * subset_percentage)

    train_subset_indices = torch.randperm(len(training_dataset))[:train_subset_size]
    val_subset_indices = torch.randperm(len(validation_dataset))[:val_subset_size]

    train_subset = torch.utils.data.Subset(training_dataset, train_subset_indices)
    val_subset = torch.utils.data.Subset(validation_dataset, val_subset_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda batch: custom_collate(batch, phoneme_vocab),
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda batch: custom_collate(batch, phoneme_vocab),
    )

    wandb.init(
        project="VALLR",
        config={
            "learning_rate": lr,
            "architecture": "VALLR",
            "dataset": "Custom Dataset",
            "epochs": epochs,
        },
    )

    model.to(device)

    for epoch in range(epochs):
        epoch_train_loss, epoch_train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, phoneme_vocab
        )

        epoch_val_loss, epoch_val_acc = validate_one_epoch(
            model, val_loader, criterion, device, phoneme_vocab
        )

        current_lr = scheduler.get_last_lr()[0]
        log_all_metrics(
            epoch + 1,
            epochs,
            epoch_train_loss,
            epoch_train_acc,
            epoch_val_loss,
            epoch_val_acc,
            current_lr,
        )

        scheduler.step(epoch_val_loss)
        save_model(model, save_model_path)

    wandb.finish()

