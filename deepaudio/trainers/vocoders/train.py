import argparse

import torch

from deepaudio.trainers.vocoders.gan.gan_vocoder_trainer import GANVocoderTrainer
from deepaudio.trainers.vocoders.diffusion.diffusion_vocoder_trainer import DiffusionVocoderTrainer
from deepaudio.trainers.vocoders.vocoder_trainer import VocoderTrainer
from deepaudio.utils.hparams import HParams


def build_trainer(args: argparse.Namespace, hparams: HParams) -> VocoderTrainer:
    supported_trainer = {
        "GANVocoder": GANVocoderTrainer,
        "DiffusionVocoder": DiffusionVocoderTrainer,
    }

    trainer_class = supported_trainer[hparams.model_type]
    trainer = trainer_class(args, hparams)
    return trainer

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train a vocoder model")
    parser.add_argument("--hparams", default="hparams.json", help="json files for configurations", required=True)
    parser.add_argument("--exp_name", type=str, default="exp_name", help="experiment name")
    parser.add_argument("--checkpoint", type=str, help="checkpoint to resume")
    parser.add_argument("--log_level", default="warning", help="logging level")
    args = parser.parse_args()
    return args

def main():
    torch.autograd.set_detect_anomaly(True)
    args = parse_args()
    hparams = HParams.from_json_file(args.hparams)
    # Override exp_name
    if "exp_name" in args:
        hparams.exp_name = args.exp_name

    # Data Augmentation
    """
    if hparams.preprocess.data_augment:
        new_datasets_list = []
        for dataset in hparams.preprocess.data_augment:
            new_datasets = [
                # f"{dataset}_pitch_shift",
                # f"{dataset}_formant_shift",
                f"{dataset}_equalizer",
                f"{dataset}_time_stretch",
            ]
            new_datasets_list.extend(new_datasets)
        hparams.dataset.extend(new_datasets_list)
    """

    # CUDA settings
    # cuda_relevant()

    # Build trainer
    trainer = build_trainer(args, hparams)

    trainer.train_loop()


if __name__ == "__main__":
    main()
