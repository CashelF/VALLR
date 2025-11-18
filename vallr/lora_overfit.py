"""Utilities to LoRA-adapt VideoMAE and overfit on a single phoneme-labeled video.

This module provides a tiny dataset wrapper for a single video + phoneme
annotation JSON, a lightweight LoRA implementation for `VideoMAEModel`, and a
convenience training loop aimed at quickly overfitting to a single example to
validate the pipeline end-to-end.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import VideoMAEConfig, Wav2Vec2Config
from Data.preprocess import preprocess_video_with_mouth_crop
from vallr.inference import load_finetuned_model

from config import get_vocab
from Models.VALLR import VALLR
import cv2 


Tensor = torch.Tensor


class LoRALinear(nn.Module):
    """Minimal LoRA adaptor for ``nn.Linear`` layers.

    The wrapped linear's parameters are frozen while the low-rank matrices are
    trainable. Only the residual term (B @ A @ x) is trained, leaving the
    original weight intact.
    """

    def __init__(self, base_layer: nn.Linear, rank: int = 8, alpha: int = 16) -> None:
        super().__init__()
        self.base = base_layer
        self.rank = rank
        self.scaling = alpha / rank

        self.lora_A = nn.Linear(base_layer.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base_layer.out_features, bias=False)

        # Freeze the base weights
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        self.reset_parameters()

    @property
    def weight(self) -> nn.Parameter:
        """Expose the frozen base weight for compatibility with F.linear calls."""

        return self.base.weight

    @property
    def bias(self) -> nn.Parameter | None:
        """Expose the frozen base bias for compatibility with F.linear calls."""

        return self.base.bias

    def reset_parameters(self) -> None:
        # nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return self.base(x) + self.lora_B(self.lora_A(x)) * self.scaling


def _replace_linear_with_lora(module: nn.Module, rank: int, alpha: int) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha))
        else:
            _replace_linear_with_lora(child, rank, alpha)


def apply_videomae_lora(videomae: nn.Module, rank: int = 8, alpha: int = 16) -> None:
    """Freeze VideoMAE and insert LoRA adapters into all linear submodules."""

    for param in videomae.parameters():
        param.requires_grad = False
    _replace_linear_with_lora(videomae, rank=rank, alpha=alpha)


@dataclass
class SingleVideoSample:
    video_path: Path
    phoneme_ids: List[int]
    aligned_phonemes: List[Dict]  # each has phoneme_id, start, end, etc.

    @staticmethod
    def from_json(annotation_path: Path) -> "SingleVideoSample":
        with open(annotation_path, "r") as fp:
            payload = json.load(fp)
        return SingleVideoSample(
            video_path=Path(payload["video_path"]),
            phoneme_ids=list(payload["phoneme_ids"]),
            aligned_phonemes=list(payload["aligned_phonemes"]),
        )



class SingleVideoPhonemeDataset(Dataset):
    """Repeats a single video + phoneme annotation for overfitting experiments."""

    def __init__(
        self,
        annotation_path: str,
        num_frames: int = 16,
        frame_size: Tuple[int, int] = (224, 224),
        repeats: int = 32,
        max_target_length: int | None = None,
        target_fps: float | None = 25.0,
        margin: float = 1.6,
        min_conf: float = 0.5,
        ema: float = 0.6,
        model_selection: int = 0,
        debug_save_path: str | None = None,
    ) -> None:
        self.sample = SingleVideoSample.from_json(Path(annotation_path))
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.repeats = max(1, repeats)
        self.max_target_length = max_target_length
        self.target_fps = target_fps
        self.margin = margin
        self.min_conf = min_conf
        self.ema = ema
        self.model_selection = model_selection
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(self.frame_size, antialias=True),
            ]
        )

        self.video_frames, self.video_fps = self._load_and_preprocess_video(self.sample.video_path)

        if debug_save_path is not None:
            self._save_debug_video(debug_save_path)

    def __len__(self) -> int:
        return self.repeats

    def __getitem__(self, _: int) -> Tuple[Tensor, Tensor]:
        video_np, (t_start, t_end) = self._load_and_sample_video()
        # (T, H, W, C) -> (T, C, H, W)
        video_tensor = torch.stack([self.transform(frame) for frame in video_np])

        # 1) take only phonemes overlapping this window
        label_ids = self._slice_targets_by_time(t_start, t_end)
        # 2) optionally downsample to max_target_length (e.g. 16)
        label_ids = self._downsample_targets(label_ids)

        label_tensor = torch.tensor(label_ids, dtype=torch.long)
        return video_tensor, label_tensor


    def _load_and_preprocess_video(self, video_path: Path) -> Tuple[np.ndarray, float]:
        return preprocess_video_with_mouth_crop(
            video_path,
            frame_size=self.frame_size,
            target_fps=self.target_fps,
            margin=self.margin,
            min_conf=self.min_conf,
            ema=self.ema,
            model_selection=self.model_selection,
        )

    def _load_and_sample_video(self) -> Tuple[np.ndarray, Tuple[float, float]]:
        video_np_full = self.video_frames
        frame_count = len(video_np_full)
        fps = self.video_fps

        if frame_count <= self.num_frames:
            # Not enough frames: take all and pad by repeating last frame
            indices = np.arange(frame_count)
            if frame_count < self.num_frames:
                pad = np.full(self.num_frames - frame_count, frame_count - 1, dtype=int)
                indices = np.concatenate([indices, pad])
            start_idx = 0
        else:
            # Random contiguous window
            start_idx = np.random.randint(0, frame_count - self.num_frames + 1)
            indices = np.arange(start_idx, start_idx + self.num_frames)

        video_np = video_np_full[indices]

        frame_start_t = start_idx / fps
        frame_end_t = (start_idx + self.num_frames) / fps

        return video_np, (frame_start_t, frame_end_t)
    
    def _slice_targets_by_time(self, t_start: float, t_end: float) -> List[int]:
        ids: List[int] = []
        for p in self.sample.aligned_phonemes:
            if p["end"] > t_start and p["start"] < t_end:
                ids.append(int(p["phoneme_id"]))

        # If no overlap, just return a single blank-ish phoneme instead of the whole sequence
        if not ids:
            # E.g. first phoneme, or a dedicated silence/blank
            ids = [self.sample.aligned_phonemes[0]["phoneme_id"]]

        return ids


    
    def _downsample_targets(self, phoneme_ids: List[int]) -> List[int]:
        if self.max_target_length is None or len(phoneme_ids) <= self.max_target_length:
            return phoneme_ids

        indices = np.linspace(0, len(phoneme_ids) - 1, self.max_target_length).astype(int)
        return [phoneme_ids[i] for i in indices]
    
    def _save_debug_video(self, out_path: str) -> None:
        """Dump the preprocessed mouth-cropped video to an MP4 for inspection."""
        frames = self.video_frames  # (T, H, W, C), assumed RGB uint8 or float
        if frames is None or len(frames) == 0:
            print("No frames to save for debug video.")
            return

        h, w, c = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, float(self.video_fps), (w, h))

        for frame in frames:
            frame_np = frame
            if frame_np.dtype != np.uint8:
                frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)

            # OpenCV expects BGR
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)

        writer.release()
        print(f"[debug] Saved preprocessed video to {out_path}")



def collate_single_video(batch: Iterable[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, List[Tensor]]:
    videos, labels = zip(*batch)
    stacked_videos = torch.stack(videos)
    label_tensors = [label.clone() for label in labels]
    return stacked_videos, label_tensors


def build_lora_overfit_model(
    phoneme_vocab: Dict[str, int], lora_rank: int, lora_alpha: int, num_frames: int
) -> VALLR:
    videomae_config = VideoMAEConfig(num_frames=num_frames)
    wav2vec_config = Wav2Vec2Config()
    wav2vec_config.vocab_size = len(phoneme_vocab)

    # model = VALLR(
    #     videomae_config=videomae_config,
    #     wav2vec_config=wav2vec_config,
    #     adapter_dim=256,
    # )
    model = load_finetuned_model('VALLR.path', torch.device('cuda'), 'V1', phoneme_vocab)
    # for name, p in model.named_parameters():
    #     p.requires_grad = False
    apply_videomae_lora(model.videomae, rank=lora_rank, alpha=lora_alpha)
    return model


def train_single_video_lora(
    annotation_path: str,
    device: torch.device,
    epochs: int = 50,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lr: float = 1e-3,
    batch_size: int = 2,
    repeats: int = 64,
    num_frames: int = 16,
    save_path: str = "checkpoints/lora_single_video.pt",
) -> None:
    phoneme_vocab = get_vocab()
    dataset = SingleVideoPhonemeDataset(
        annotation_path=annotation_path,
        num_frames=num_frames,
        repeats=repeats,
        max_target_length=num_frames // 2, # Encoder downsamples time by 2
        # debug_save_path="debug_preprocessed_mouthcrop.mp4",
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_single_video,
    )

    model = build_lora_overfit_model(
        phoneme_vocab=phoneme_vocab,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        num_frames=num_frames,
    )
    model.to(device)

    criterion = nn.CTCLoss(blank=phoneme_vocab["<pad>"], reduction="mean", zero_infinity=True)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.0
    )

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for videos, labels in dataloader:
            videos = videos.to(device).float()
            labels = [label.to(device) for label in labels]

            # optimizer.zero_grad()
            logits, _ = model(videos)
            log_probs = logits.log_softmax(dim=-1).transpose(0, 1)

            batch_size_val = log_probs.size(1)
            input_lengths = torch.full(
                size=(batch_size_val,), fill_value=log_probs.size(0), dtype=torch.long, device=device
            )
            target_lengths = torch.tensor([label.size(0) for label in labels], dtype=torch.long, device=device)

            if input_lengths.min() < target_lengths.max():
                print(
                    f"Skipping batch: input lengths {input_lengths.min().item()} < target lengths {target_lengths.max().item()}"
                )
                continue
            
            with torch.no_grad():
                # log_probs: (T, B, V)
                pred_ids = log_probs.argmax(dim=-1)  # (T, B)

                # Take the first item in the batch for inspection
                pred_seq_b0 = pred_ids[:, 0].detach().cpu().tolist()
                target_seq_b0 = labels[0].detach().cpu().tolist()

                # Build inverse vocab: id -> phoneme string
                id2phoneme = {idx: ph for ph, idx in phoneme_vocab.items()}
                blank_id = phoneme_vocab["<pad>"]

                def ids_to_phonemes(seq):
                    return [id2phoneme.get(i, f"<unk:{i}>") for i in seq]

                # Simple CTC-style collapse (remove repeats + blanks)
                def ctc_collapse(seq, blank=blank_id):
                    out = []
                    prev = None
                    for i in seq:
                        if i == blank:
                            continue
                        if i == prev:
                            continue
                        out.append(i)
                        prev = i
                    return out

                collapsed_pred_b0 = ctc_collapse(pred_seq_b0)

                # Map to phoneme strings
                raw_pred_phonemes     = ids_to_phonemes(pred_seq_b0)
                collapsed_pred_phonemes = ids_to_phonemes(collapsed_pred_b0)
                target_phonemes       = ids_to_phonemes(target_seq_b0)

                print("\n=== DEBUG: batch 0 ===")
                print("raw pred ids (per timestep):", pred_seq_b0)
                print("raw pred phonemes:          ", raw_pred_phonemes)
                print()
                print("collapsed pred ids:         ", collapsed_pred_b0)
                print("collapsed pred phonemes:    ", collapsed_pred_phonemes)
                print()
                print("target ids:                 ", target_seq_b0)
                print("target phonemes:            ", target_phonemes)
                print("======================\n")


            loss = criterion(log_probs, torch.cat(labels), input_lengths, target_lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / max(1, len(dataloader))
        print(f"Epoch {epoch + 1}/{epochs} - training loss: {avg_loss:.4f}")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Saved LoRA-overfit checkpoint to {save_path}")
