"""Streamed inference script for arbitrary-length videos."""

from __future__ import annotations

import argparse
from typing import Generator, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from decord import VideoReader, cpu

from config import get_vocab
from vallr.inference import load_finetuned_model


Tensor = torch.Tensor


def iter_video_chunks(
    video_path: str,
    chunk_size: int,
    stride: Optional[int] = None,
    num_threads: int = 4,
    frame_size: Tuple[int, int] = (224, 224),
) -> Generator[Tensor, None, None]:
    """Yield sliding-window chunks of frames from ``video_path`` prepared for inference.

    Each yielded tensor has shape ``(1, chunk_size, C, H, W)``. The final chunk is
    padded by repeating the last available frame when the total frame count is not
    an exact multiple of ``chunk_size``.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")

    if stride is None:
        stride = chunk_size
    if stride <= 0:
        raise ValueError("stride must be a positive integer")

    try:
        video_reader = VideoReader(video_path, ctx=cpu(0), num_threads=num_threads)
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError(f"Unable to open video '{video_path}': {exc}") from exc

    total_frames = len(video_reader)
    if total_frames == 0:
        raise RuntimeError(f"Video '{video_path}' does not contain any frames")

    frames: List[np.ndarray] = []
    for idx in range(total_frames):
        frame = video_reader[idx].asnumpy()
        frames.append(preprocess_frame(frame, frame_size))

    # Always include the final window so that the tail of the clip is covered.
    window_starts = list(range(0, max(total_frames - chunk_size + 1, 1), stride))
    last_start = max(total_frames - chunk_size, 0)
    if window_starts[-1] != last_start:
        window_starts.append(last_start)

    for start_idx in window_starts:
        window_frames = frames[start_idx : start_idx + chunk_size]
        while len(window_frames) < chunk_size:
            window_frames.append(window_frames[-1])

        chunk_np = np.stack(window_frames)
        chunk_tensor = torch.from_numpy(chunk_np).unsqueeze(0).float()
        yield chunk_tensor


def preprocess_frame(frame: np.ndarray, frame_size: Tuple[int, int]) -> np.ndarray:
    """Apply VALLR V1 preprocessing to a single frame.

    Steps mirror the training/inference path: resize to 224x224, convert to float
    in [0, 1], and normalize using VideoMAE-style statistics.
    """

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    resized = cv2.resize(frame, frame_size, interpolation=cv2.INTER_AREA)
    float_frame = resized.astype(np.float32) / 255.0
    normalized = (float_frame - mean) / std
    chw = np.transpose(normalized, (2, 0, 1))
    return chw


def decode_logits(
    logits: Tensor,
    reverse_vocab: Mapping[int, str],
    suppress_repeats: bool = True,
) -> List[str]:
    """Convert model logits into a sequence of phoneme tokens."""

    predicted_indices = torch.argmax(logits, dim=-1).squeeze(0).tolist()
    decoded: List[str] = []
    previous_token: Optional[str] = None

    for index in predicted_indices:
        token = reverse_vocab.get(index)
        if token is None or token == "<pad>":
            continue
        if suppress_repeats and token == previous_token:
            continue
        decoded.append(token)
        previous_token = token

    return decoded


def translate_phonemes_with_llm(phonemes: Sequence[str]) -> str:
    """Placeholder hook for connecting an LLM based phoneme translator.

    Replace this function body with the integration to your preferred LLM. When the
    ``--translate`` CLI flag is used the script will call this function for each
    decoded sequence.
    """

    raise NotImplementedError("LLM translation is not implemented. Integrate your LLM here.")


def run_full_clip_inference(
    model: torch.nn.Module,
    video_path: str,
    device: torch.device,
    reverse_vocab: Mapping[int, str],
    frame_size: Tuple[int, int],
    debug: bool,
) -> List[str]:
    """Load the entire clip, run one forward pass, and decode phonemes."""

    try:
        video_reader = VideoReader(video_path, ctx=cpu(0), num_threads=4)
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError(f"Unable to open video '{video_path}': {exc}") from exc

    frames: List[np.ndarray] = []
    for idx in range(len(video_reader)):
        frame = video_reader[idx].asnumpy()
        frames.append(preprocess_frame(frame, frame_size))

    if not frames:
        raise RuntimeError(f"Video '{video_path}' does not contain any frames")

    clip = torch.from_numpy(np.stack(frames)).unsqueeze(0).float().to(device)

    with torch.no_grad():
        logits, _ = model(clip)

    if debug:
        predicted_indices = torch.argmax(logits, dim=-1).squeeze(0).tolist()
        print("Pred indices:", predicted_indices[:50])
        print("Pred tokens:", [reverse_vocab.get(i) for i in predicted_indices[:50]])

    return decode_logits(logits, reverse_vocab)


def run_streaming_inference(
    model_path: str,
    model_version: str,
    video_path: str,
    device: torch.device,
    chunk_size: int,
    stride: int,
    translate: bool,
    no_stream: bool,
    debug: bool,
    frame_size: Tuple[int, int],
) -> None:
    """Run streamed inference on ``video_path`` and print phoneme predictions."""

    phoneme_vocab = get_vocab()
    model = load_finetuned_model(model_path, device, model_version, phoneme_vocab)
    reverse_vocab = {value: key for key, value in phoneme_vocab.items()}

    if no_stream:
        phonemes = run_full_clip_inference(
            model=model,
            video_path=video_path,
            device=device,
            reverse_vocab=reverse_vocab,
            frame_size=frame_size,
            debug=debug,
        )
        print(f"Full clip: {' '.join(phonemes) if phonemes else '[no phonemes]'}")
        return

    all_phoneme_sequences: List[List[str]] = []

    with torch.no_grad():
        for chunk_idx, chunk_tensor in enumerate(
            iter_video_chunks(video_path, chunk_size, stride=stride, frame_size=frame_size), start=1
        ):
            chunk_tensor = chunk_tensor.to(device)
            logits, _ = model(chunk_tensor)

            if debug and chunk_idx == 1:
                predicted_indices = torch.argmax(logits, dim=-1).squeeze(0).tolist()
                print("Pred indices:", predicted_indices[:50])
                print("Pred tokens:", [reverse_vocab.get(i) for i in predicted_indices[:50]])

            phoneme_sequence = decode_logits(logits, reverse_vocab)
            all_phoneme_sequences.append(phoneme_sequence)

    for idx, phonemes in enumerate(all_phoneme_sequences, start=1):
        print(f"Chunk {idx}: {' '.join(phonemes) if phonemes else '[no phonemes]'}")

        if translate:
            try:
                sentence = translate_phonemes_with_llm(phonemes)
            except NotImplementedError as exc:
                print(f"  Translation skipped: {exc}")
                continue
            print(f"  Translation: {sentence}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream inference over a video file")
    parser.add_argument("model_path", help="Path to the fine-tuned model checkpoint")
    parser.add_argument("video_path", help="Path to the video file to process")
    parser.add_argument(
        "--version",
        default="V1",
        choices=["V1", "V2"],
        help="Model variant to load",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to run inference on",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=16,
        help="Number of frames per inference pass (matches V1 training sequence length)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Stride (in frames) between sliding windows; defaults to chunk size (no overlap)",
    )
    parser.add_argument(
        "--translate",
        action="store_true",
        help="Attempt to translate phonemes into text using an LLM hook",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming and run a single forward pass over the full clip",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print raw predicted indices and tokens for the first decoded output",
    )
    parser.add_argument(
        "--frame-size",
        type=int,
        default=224,
        help="Spatial resolution for preprocessing (height=width)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    device = torch.device(args.device)
    run_streaming_inference(
        model_path=args.model_path,
        model_version=args.version,
        video_path=args.video_path,
        device=device,
        chunk_size=args.chunk_size,
        stride=args.stride if args.stride is not None else args.chunk_size,
        translate=args.translate,
        no_stream=args.no_stream,
        debug=args.debug,
        frame_size=(args.frame_size, args.frame_size),
    )


if __name__ == "__main__":
    main()
