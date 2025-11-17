"""Inference utilities mirroring the original main.py behaviour."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from decord import VideoReader, cpu

# from Models.ML_VALLR import ML_VALLR
from Models.VALLR import VALLR
from transformers import VideoMAEConfig, Wav2Vec2Config


Tensor = torch.Tensor


def load_videos(video_path: str, num_frames: int = 16, frame_size: Tuple[int, int] = (224, 224)) -> Optional[Tensor]:
    """Load a fixed number of frames from disk and return a tensor shaped for inference."""

    del frame_size  # preserved for signature compatibility; resizing handled elsewhere if needed.

    try:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=4)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Error loading video: {video_path}. Error: {exc}")
        return None

    frame_count = len(vr)
    if frame_count < num_frames:
        print(f"Warning: Not enough frames in video {video_path}. Skipping.")
        return None

    sample_indices = np.linspace(0, frame_count - 1, num_frames).astype(int)

    frames: List[np.ndarray] = []
    for idx in sample_indices:
        frame = vr[idx].asnumpy()
        frame = np.transpose(frame, (2, 0, 1))
        frames.append(frame)

    if len(frames) < num_frames:
        return None

    video_np = np.array(frames)
    video_tensor = torch.tensor(video_np).float()
    video_tensor = video_tensor.unsqueeze(0)
    return video_tensor


def _build_model(version: str, phoneme_vocab: Dict[str, int]) -> torch.nn.Module:
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


def load_finetuned_model(model_path: str, device: torch.device, version: str, vocab: Dict[str, int]) -> torch.nn.Module:
    """Load a fine-tuned checkpoint onto the requested device."""

    phoneme_vocab = vocab
    model = _build_model(version, phoneme_vocab)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded from {model_path}")
    return model


def run_inference(
    model_path: str,
    version: str,
    video_path: str,
    device: torch.device,
    phoneme_vocab: Dict[str, int],
    beam_width: int = 3,
) -> Tuple[List[List[str]], torch.Tensor]:
    """Run inference and return decoded phoneme predictions and intermediate features."""

    del beam_width  # Retained for signature compatibility; beam search not implemented.

    video_inputs = load_videos(video_path, num_frames=16)
    if video_inputs is None:
        print("Error: Failed to load video frames.")
        return []  # type: ignore[return-value]

    model = load_finetuned_model(model_path, device, version, phoneme_vocab)
    reverse_vocab = {v: k for k, v in phoneme_vocab.items()}

    model.eval()

    with torch.no_grad():
        video_inputs = video_inputs.to(device).float()
        logits, feats = model(video_inputs)
        predicted_indices = torch.argmax(logits, dim=-1)

        predicted_phonemes: List[List[str]] = []
        for pred_idx_seq in predicted_indices:
            phoneme_seq = [reverse_vocab[idx] for idx in pred_idx_seq.tolist() if idx in reverse_vocab]
            predicted_phonemes.append(phoneme_seq)

    return predicted_phonemes, feats

