"""Command-line interface for training and inference with VALLR."""

import torch

from config import get_vocab, load_args
from vallr.inference import run_inference
from vallr.training import train
from vallr.lora_overfit import train_single_video_lora


def main() -> None:
    args = load_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    video_path = args.videos_root
    batch_size = args.batch_size
    num_workers = args.num_workers
    epochs = args.epochs
    save_model_path = args.save_model_path
    sample_size = args.sample_size
    version = args.version
    vocab = get_vocab()

    if args.mode == "train":
        print("Training")
        train(
            device,
            version,
            video_path,
            batch_size,
            num_workers,
            epochs,
            save_model_path,
            sample_size,
            vocab,
        )
    elif args.mode == "infer":
        print("Inferences", run_inference(save_model_path, version, video_path, device, vocab))
    elif args.mode == "lora_overfit":
        print("Lora Overfit")
        train_single_video_lora('train_phonemes.json', torch.device('cuda'))
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
