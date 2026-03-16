"""Example: Training a HiFi-GAN vocoder."""

import torch
from soniq.training import FabricTrainer
from soniq.models.vocoders.hifigan import HifiGAN, HifiGANConfig
from soniq.datasets import BaseDataset, BaseCollator, build_dataloader


def main():
    # Configuration
    config = HifiGANConfig(
        inter_channels=128,
        upsample_rates=[8, 8, 4, 2],
        upsample_initial_channel=512,
    )

    # Model
    model = HifiGAN(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Dataset (update paths to your data)
    dataset = BaseDataset(
        metadata_path="data/train.json",
        feature_dirs={"mel": "data/mels", "wav": "data/wavs"},
        sample_rate=24000,
    )

    # DataLoader
    dataloader = build_dataloader(
        dataset,
        BaseCollator(),
        batch_size=16,
        num_workers=4,
    )

    # Trainer
    trainer = FabricTrainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        max_epochs=100,
    )

    # Training step
    def train_step(model, batch):
        mel = batch["mel"]
        audio = batch["wav"]
        generated = model(mel)
        loss = torch.nn.functional.l1_loss(generated, audio)
        return loss

    # Train
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    trainer.fit(
        model,
        dataloader,
        optimizer=optimizer,
        train_step_fn=train_step,
    )

    print("Training completed!")


if __name__ == "__main__":
    main()
