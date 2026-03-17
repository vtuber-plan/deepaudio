#!/usr/bin/env python
# coding=utf-8
"""
Test Accelerate distributed training backend.

Usage:
    accelerate launch tests/test_accelerate_backend.py
    accelerate launch --num_processes 2 tests/test_accelerate_backend.py
"""

import os
import sys
import tempfile
import shutil

import torch
from torch.utils.data import Dataset, DataLoader

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from accelerate import Accelerator


def test_accelerate_backend():
    """Test Accelerate distributed backend."""
    accelerator = Accelerator()

    # Print backend info
    if accelerator.is_main_process:
        print("=" * 60)
        print("Accelerate Backend Information")
        print("=" * 60)

    accelerator.print(f"[Rank {accelerator.process_index}] Device: {accelerator.device}")
    accelerator.print(f"[Rank {accelerator.process_index}] Local rank: {accelerator.local_process_index}")
    accelerator.print(f"[Rank {accelerator.process_index}] Num processes: {accelerator.num_processes}")
    accelerator.print(f"[Rank {accelerator.process_index}] Distributed type: {accelerator.distributed_type}")
    accelerator.print(f"[Rank {accelerator.process_index}] Mixed precision: {accelerator.mixed_precision}")

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        print("\n" + "=" * 60)
        print("Testing distributed operations")
        print("=" * 60)

    # Test simple model
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Create dummy dataset
    class DummyDataset(Dataset):
        def __len__(self):
            return 100
        def __getitem__(self, idx):
            return {"x": torch.randn(10), "y": torch.randn(10)}

    dataset = DummyDataset()
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Prepare with accelerator
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    accelerator.print(f"[Rank {accelerator.process_index}] Model prepared on {next(model.parameters()).device}")

    # Run a few training steps
    model.train()
    for i, batch in enumerate(dataloader):
        if i >= 5:
            break

        optimizer.zero_grad()
        output = model(batch["x"])
        loss = torch.nn.functional.mse_loss(output, batch["y"])
        accelerator.backward(loss)
        optimizer.step()

        accelerator.print(f"[Rank {accelerator.process_index}] Step {i}: loss = {loss.item():.4f}")

    accelerator.wait_for_everyone()

    # Test gather
    tensor = torch.tensor([accelerator.process_index], device=accelerator.device)
    gathered = accelerator.gather(tensor)
    if accelerator.is_main_process:
        print(f"\nGathered tensors: {gathered.tolist()}")

    # Test all_reduce
    tensor = torch.tensor([1.0], device=accelerator.device)
    reduced = accelerator.reduce(tensor, reduction="mean")
    accelerator.print(f"[Rank {accelerator.process_index}] All reduce mean: {reduced.item():.4f}")

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        print("\n" + "=" * 60)
        print("Accelerate backend test PASSED!")
        print("=" * 60)

    return True


if __name__ == "__main__":
    test_accelerate_backend()