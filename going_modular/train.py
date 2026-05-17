"""
Trains a PyTorch image classification model using device-agnostic code.
Supports resuming from a checkpoint with metadata validation.
"""

import os
import json
import torch
import argparse
import data_setup, engine, model_builder, utils

from torchvision import transforms

# Step 1: Add --resume_checkpoint as an optional path argument via argparse
parser = argparse.ArgumentParser(description="Train a PyTorch image classification model.")
parser.add_argument("--resume_checkpoint", 
                    type=str, 
                    default=None, 
                    help="Path to a checkpoint file (.pth) to resume training from.")
# Parse known args so it works cleanly both in terminal and Jupyter environments
args, _ = parser.parse_known_args()

# Setup hyperparameters
NUM_EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

# Setup directories
train_dir = "data/pizza_steak_sushi/train"
test_dir = "data/pizza_steak_sushi/test"

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create transforms
data_transform = transforms.Compose([
  transforms.Resize((64, 64)),
  transforms.ToTensor()
])

# Create DataLoaders with help from data_setup.py
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=data_transform,
    batch_size=BATCH_SIZE
)

# Create model with help from model_builder.py
model = model_builder.TinyVGG(
    input_shape=3,
    hidden_units=HIDDEN_UNITS,
    output_shape=len(class_names)
).to(device)

# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

# Variables to track training progress
start_epoch = 0
previous_history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

# Step 2 & 3: Load model, optimizer, and metadata if checkpoint path is provided
if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
    print(f"[INFO] Loading checkpoint from: {args.resume_checkpoint}")
    checkpoint = torch.load(args.resume_checkpoint, map_location=device)
    
    # Hints: Validate that resumed checkpoints match the selected model architecture
    checkpoint_metadata = checkpoint.get("model_metadata", {})
    current_metadata = model.metadata
    
    if checkpoint_metadata.get("architecture") != current_metadata.get("architecture") or \
       checkpoint_metadata.get("hidden_units") != current_metadata.get("hidden_units"):
        raise ValueError(
            f"Architecture mismatch! Checkpoint: {checkpoint_metadata}, Current Model: {current_metadata}"
        )
        
    print("[INFO] Checkpoint architecture validated successfully.")
    
    # Load state dicts
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "scheduler_state_dict" in checkpoint and scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
    # Restore starting epoch and history
    start_epoch = checkpoint["epoch"] + 1
    if "history" in checkpoint:
        previous_history = checkpoint["history"]
        
    print(f"[INFO] Resuming training from epoch {start_epoch + 1}")
    print(f"[INFO] Reported resumed epoch range: {start_epoch + 1} to {NUM_EPOCHS}")

print("\n--- Experiment Configuration ---")
print(f"Model Metadata: {model.metadata}")
print(f"Optimizer: {optimizer.__class__.__name__} (Initial LR: {LEARNING_RATE})")
print(f"Scheduler: {scheduler.__class__.__name__} (Step Size: {scheduler.step_size}, Gamma: {scheduler.gamma})")
print("--------------------------------\n")

config_log = {
    "model_metadata": model.metadata,
    "hyperparameters": {
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "initial_learning_rate": LEARNING_RATE
    },
    "optimizer": optimizer.__class__.__name__,
    "scheduler": {
        "type": scheduler.__class__.__name__,
        "step_size": scheduler.step_size,
        "gamma": scheduler.gamma
    }
}

os.makedirs("model_logs", exist_ok=True)
with open("model_logs/experiment_config.json", "w") as f:
    json.dump(config_log, f, indent=4)
print("[INFO] Combined training configuration saved to 'model_logs/experiment_config.json'")

# Step 4: Adjust total epochs and run training loop
remaining_epochs = NUM_EPOCHS - start_epoch

if remaining_epochs > 0:
    # Start training with help from engine.py
    new_results = engine.train(model=model,
                               train_dataloader=train_dataloader,
                               test_dataloader=test_dataloader,
                               loss_fn=loss_fn,
                               optimizer=optimizer,
                               epochs=remaining_epochs,
                               device=device)
    
    # Append new metrics to the existing history
    for key in previous_history:
        previous_history[key].extend(new_results[key])
else:
    print("[INFO] Model already trained for the specified number of epochs.")

# Step 5: Save a new checkpoint after resumption
checkpoint_save_path = "models/checkpoint_epoch_latest.pth"
torch.save({
    "epoch": NUM_EPOCHS - 1,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
    "model_metadata": model.metadata,
    "history": previous_history
}, checkpoint_save_path)

print(f"[INFO] New checkpoint saved to '{checkpoint_save_path}'")

# Save final standalone model weights with help from utils.py
utils.save_model(model=model,
                 target_dir="models",
                 model_name="06_going_modular_script_mode_tinyvgg_model.pth")