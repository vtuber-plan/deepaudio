# Training API

This page documents the training utilities in Soniq using Lightning Fabric.

## FabricTrainer

Base trainer class for training models with Lightning Fabric.

### Usage

```python
from soniq.training import FabricTrainer

# Create trainer
trainer = FabricTrainer(
    accelerator="gpu",
    strategy="ddp",
    devices=4,
    precision="16-mixed",
    max_epochs=100,
    gradient_accumulation_steps=1,
    gradient_clip_val=1.0,
    loggers=["tensorboard"],
    default_root_dir="./logs",
)

# Train model
trainer.fit(
    model,
    train_dataloader,
    val_dataloader,
    optimizer=optimizer,
    scheduler=scheduler,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `accelerator` | str | "auto" | Hardware accelerator ("cpu", "gpu", "mps") |
| `strategy` | str | "auto" | Training strategy ("auto", "ddp", "fsdp") |
| `devices` | int/list/str | None | Number of devices or device IDs |
| `num_nodes` | int | 1 | Number of nodes for distributed training |
| `precision` | str | "32-true" | Precision mode ("32-true", "16-mixed", "bf16-mixed") |
| `max_epochs` | int | 100 | Maximum number of epochs |
| `max_steps` | int | None | Maximum number of steps |
| `gradient_accumulation_steps` | int | 1 | Gradient accumulation steps |
| `gradient_clip_val` | float | None | Gradient clipping value |
| `loggers` | list | ["tensorboard"] | List of loggers |
| `default_root_dir` | str | "./logs" | Root directory for logs and checkpoints |

## Training Loop

### Custom Training Step

```python
def train_step(model, batch):
    # Forward pass
    output = model(batch["input"])
    target = batch["target"]

    # Compute loss
    loss = torch.nn.functional.mse_loss(output, target)

    return loss

# Train with custom step
trainer.fit(
    model,
    train_dataloader,
    optimizer=optimizer,
    train_step_fn=train_step,
)
```

### Custom Validation

```python
def validate_fn(model, batch):
    with torch.no_grad():
        output = model(batch["input"])
        target = batch["target"]
        loss = torch.nn.functional.mse_loss(output, target)
    return loss

# Train with validation
trainer.fit(
    model,
    train_dataloader,
    val_dataloader,
    optimizer=optimizer,
    train_step_fn=train_step,
    validate_fn=validate_fn,
)
```

## Checkpointing

### Save Checkpoint

```python
trainer.save_checkpoint(
    path="checkpoints/epoch-001.pt",
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    extra_state={"epoch": epoch, "metrics": metrics},
)
```

### Load Checkpoint

```python
state = trainer.load_checkpoint(
    path="checkpoints/epoch-001.pt",
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
)
```

### Automatic Checkpointing

The trainer automatically saves checkpoints:
- At the end of each epoch
- When training is interrupted

Checkpoints are saved to `default_root_dir/checkpoints/`.

## Logging

### TensorBoard

```python
trainer = FabricTrainer(
    loggers=["tensorboard"],
    default_root_dir="./logs",
)

# Log metrics in training step
def train_step(model, batch):
    loss = compute_loss(model, batch)
    trainer.fabric.log("train/loss", loss)
    return loss
```

### Custom Loggers

```python
from lightning.fabric.loggers import WandbLogger

trainer = FabricTrainer(
    loggers=[
        WandbLogger(project="soniq", name="experiment_1"),
    ],
)
```

## Distributed Training

### Data Parallel (DDP)

```python
trainer = FabricTrainer(
    accelerator="gpu",
    devices=4,
    strategy="ddp",
)
```

### Fully Sharded Data Parallel (FSDP)

```python
trainer = FabricTrainer(
    accelerator="gpu",
    devices=4,
    strategy="fsdp",
    precision="bf16-mixed",
)
```

## Mixed Precision

### 16-bit Mixed Precision

```python
trainer = FabricTrainer(
    precision="16-mixed",
)
```

### BFloat16 Mixed Precision

```python
trainer = FabricTrainer(
    precision="bf16-mixed",
)
```

## Gradient Accumulation

```python
trainer = FabricTrainer(
    gradient_accumulation_steps=4,  # Accumulate over 4 steps
)
```

This is equivalent to 4x larger batch size with less memory usage.
