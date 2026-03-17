"""Example: Training a HiFi-GAN vocoder using the unified Trainer."""

import torch
from soniq.training import Trainer, BaseTaskSystem, StepOutput
from soniq.models.vocoders.hifigan import HifiGAN, HifiGANConfig
from soniq.datasets import BaseDataset, BaseCollator, build_dataloader


class SimpleVocoderSystem(BaseTaskSystem):
    """Simple vocoder training system."""

    def __init__(self, config, generator):
        super().__init__(config)
        self.generator = generator

    def training_step(self, batch, batch_idx):
        mel = batch["mel"]
        audio = batch["wav"]
        generated = self.generator(mel)
        loss = torch.nn.functional.l1_loss(generated, audio)
        return StepOutput(loss=loss, metrics={"l1_loss": loss.item()})

    def configure_optimizers(self):
        return torch.optim.AdamW(self.generator.parameters(), lr=2e-4)


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

    # Create training system
    system = SimpleVocoderSystem(config={}, generator=model)

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

    # Unified Trainer (can switch between accelerate and fabric)
    trainer = Trainer(
        engine="accelerate",  # or "fabric"
        run_path="./outputs/hifigan_example",
        max_epochs=100,
        gradient_clip_val=1.0,
    )

    # Train
    trainer.fit(
        system=system,
        train_dataloader=dataloader,
    )

    print("Training completed!")


if __name__ == "__main__":
    main()