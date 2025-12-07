"""
Training script for GDSS (Graph Diffusion via the System of SDEs)

This script trains the score-based diffusion model from scratch on graph datasets.
Usage:
    python train.py --config train_qm9 --seed 42
"""

import os
import argparse
import torch
import torch.optim as optim
from tqdm import tqdm

# Add src directory to path
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from gdss.parsers.config import get_config
from gdss.utils.loader import (
    load_device,
    load_data,
    load_sde,
    load_seed,
    load_model_from_ckpt,
    load_model_optimizer,
    load_model_params,
)
from gdss.core.losses import get_sde_loss_fn
from gdss.utils.ema import ExponentialMovingAverage
from gdss.utils.logger import Logger, set_log, start_log, train_log


def train_epoch(
    model_x,
    model_adj,
    optimizer_x,
    optimizer_adj,
    ema_x,
    ema_adj,
    train_loader,
    loss_fn,
    config,
    device,
    epoch,
):
    """Train for one epoch"""
    model_x.train()
    model_adj.train()

    total_loss = 0.0
    total_loss_x = 0.0
    total_loss_adj = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.train.num_epochs}")
    for batch_idx, (x, adj) in enumerate(pbar):
        x = x.to(device)
        adj = adj.to(device)

        # Forward pass
        loss_x, loss_adj = loss_fn(model_x, model_adj, x, adj)
        loss = loss_x + loss_adj

        # Backward pass
        optimizer_x.zero_grad()
        optimizer_adj.zero_grad()
        loss.backward()

        # Gradient clipping
        if hasattr(config.train, "grad_clip"):
            torch.nn.utils.clip_grad_norm_(model_x.parameters(), config.train.grad_clip)
            torch.nn.utils.clip_grad_norm_(
                model_adj.parameters(), config.train.grad_clip
            )

        optimizer_x.step()
        optimizer_adj.step()

        # Update EMA
        if ema_x is not None:
            ema_x.update(model_x.parameters())
        if ema_adj is not None:
            ema_adj.update(model_adj.parameters())

        # Track losses
        total_loss += loss.item()
        total_loss_x += loss_x.item()
        total_loss_adj += loss_adj.item()

        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "loss_x": f"{loss_x.item():.4f}",
                "loss_adj": f"{loss_adj.item():.4f}",
            }
        )

    avg_loss = total_loss / len(train_loader)
    avg_loss_x = total_loss_x / len(train_loader)
    avg_loss_adj = total_loss_adj / len(train_loader)

    return avg_loss, avg_loss_x, avg_loss_adj


def save_checkpoint(
    epoch,
    model_x,
    model_adj,
    optimizer_x,
    optimizer_adj,
    ema_x,
    ema_adj,
    config,
    checkpoint_dir,
    params_x=None,
    params_adj=None,
    prefix="checkpoint",
):
    """Save model checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(checkpoint_dir, f"{prefix}_epoch_{epoch + 1}.pth")

    checkpoint = {
        "epoch": epoch,
        "config": config,
        "params_x": params_x if params_x is not None else config.model.x,
        "params_adj": params_adj if params_adj is not None else config.model.adj,
        "x_state_dict": model_x.state_dict(),
        "adj_state_dict": model_adj.state_dict(),
        "optimizer_x": optimizer_x.state_dict(),
        "optimizer_adj": optimizer_adj.state_dict(),
    }

    if config.train.ema > 0:
        checkpoint["ema_x"] = ema_x.state_dict()
        checkpoint["ema_adj"] = ema_adj.state_dict()

    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

    return checkpoint_path


def train(config):
    """Main training function"""

    # Set up logging
    log_folder_name, log_dir, _ = set_log(config, is_train=True)
    logger = Logger(str(os.path.join(log_dir, "train.log")), mode="a")

    logger.log("=" * 50)
    logger.log(f"Training {config.train.name}")
    start_log(logger, config)
    train_log(logger, config)
    logger.log("=" * 50)

    # Set random seed
    load_seed(config.seed)

    # Load device
    device = load_device()
    device_id = f"cuda:{device[0]}" if isinstance(device, list) else device
    logger.log(f"Using device: {device_id}")

    # Load data
    logger.log("Loading data...")
    train_loader, test_loader = load_data(config, get_graph_list=False)
    logger.log(f"Training batches: {len(train_loader)}")

    # Load SDEs
    logger.log("Setting up SDEs...")
    sde_x = load_sde(config.sde.x)
    sde_adj = load_sde(config.sde.adj)

    # Load models and optimizers
    logger.log("Initializing models...")
    params_x, params_adj = load_model_params(config)
    model_x, optimizer_x, scheduler_x = load_model_optimizer(
        params_x, config.train, device
    )
    model_adj, optimizer_adj, scheduler_adj = load_model_optimizer(
        params_adj, config.train, device
    )

    # Count parameters
    params_x = sum(p.numel() for p in model_x.parameters())
    params_adj = sum(p.numel() for p in model_adj.parameters())
    logger.log(f"Model X parameters: {params_x:,}")
    logger.log(f"Model Adj parameters: {params_adj:,}")

    # Setup EMA
    if config.train.ema > 0:
        logger.log(f"Using EMA with decay={config.train.ema}")
        ema_x = ExponentialMovingAverage(model_x.parameters(), decay=config.train.ema)
        ema_adj = ExponentialMovingAverage(
            model_adj.parameters(), decay=config.train.ema
        )
    else:
        ema_x = None
        ema_adj = None

    # Loss function
    loss_fn = get_sde_loss_fn(
        sde_x,
        sde_adj,
        train=True,
        reduce_mean=config.train.reduce_mean,
        eps=config.train.eps,
    )

    # Create checkpoint directory
    checkpoint_dir = os.path.join("./checkpoints", config.data.data)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training loop
    logger.log("Starting training...")
    best_loss = float("inf")

    for epoch in range(config.train.num_epochs):
        avg_loss, avg_loss_x, avg_loss_adj = train_epoch(
            model_x,
            model_adj,
            optimizer_x,
            optimizer_adj,
            ema_x,
            ema_adj,
            train_loader,
            loss_fn,
            config,
            device_id,
            epoch,
        )

        logger.log(
            f"Epoch {epoch + 1}/{config.train.num_epochs}: "
            f"Loss={avg_loss:.4f}, Loss_X={avg_loss_x:.4f}, Loss_Adj={avg_loss_adj:.4f}"
        )

        # Save checkpoint periodically
        if (epoch + 1) % 100 == 0:
            save_checkpoint(
                epoch,
                model_x,
                model_adj,
                optimizer_x,
                optimizer_adj,
                ema_x,
                ema_adj,
                config,
                checkpoint_dir,
                params_x=params_x,
                params_adj=params_adj,
                prefix=config.train.name,
            )

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(
                epoch,
                model_x,
                model_adj,
                optimizer_x,
                optimizer_adj,
                ema_x,
                ema_adj,
                config,
                checkpoint_dir,
                params_x=params_x,
                params_adj=params_adj,
                prefix=f"{config.train.name}_best",
            )
            logger.log(f"New best model saved with loss={best_loss:.4f}")

    # Save final checkpoint
    final_path = save_checkpoint(
        config.train.num_epochs - 1,
        model_x,
        model_adj,
        optimizer_x,
        optimizer_adj,
        ema_x,
        ema_adj,
        config,
        checkpoint_dir,
        params_x=params_x,
        params_adj=params_adj,
        prefix=f"{config.train.name}_final",
    )

    logger.log("=" * 50)
    logger.log("Training completed!")
    logger.log(f"Final checkpoint: {final_path}")
    logger.log(f"Best loss: {best_loss:.4f}")
    logger.log("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Train GDSS model")
    parser.add_argument(
        "--config",
        type=str,
        default="train_qm9",
        help="Config file name (without .yaml extension)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Load configuration
    config = get_config(args.config, args.seed)

    # Start training
    train(config)


if __name__ == "__main__":
    main()
