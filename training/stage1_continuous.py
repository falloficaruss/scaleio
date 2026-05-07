import os
import random
import sys
import time
from datetime import datetime
from typing import Dict, Optional

import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.datasets import ContinuousScaleData
from evaluation.metrics import batch_metrics, calculate_psnr, calculate_ssim
from models.c2d_isr import C2DISRFactory
from training.losses import L1Loss
from training.scheduler import WarmupCosineScheduler


class Stage1Trainer:
    """Stage 1: Continuous-scale pre-training with HIIF-L upsampler."""

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Training parameters
        self.epochs = config.get("epochs", 700)
        self.batch_size = config.get("batch_size", 16)
        self.lr_max = config.get("lr_max", 4e-4)
        self.lr_min = config.get("lr_min", 1e-6)
        self.warmup_epochs = config.get("warmup_epochs", 50)
        self.min_scale = config.get("min_scale", 1.0)
        self.max_scale = config.get("max_scale", 4.0)

        # Model parameters
        self.feature_dim = config.get("feature_dim", 96)
        self.num_hiet_blocks = config.get("num_hiet_blocks", 3)
        self.num_heads = config.get("num_heads", 8)

        # Paths
        self.data_path = config.get("data_path", "data/div2k")
        self.save_path = config.get("save_path", "checkpoints")
        self.log_path = config.get("log_path", "logs")

        # Create directories
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)

        # Initialize components
        self._setup_model()
        self._setup_data()
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_loss()
        self._setup_logging()

        # Training state
        self.current_epoch = 0
        self.best_psnr = 0.0
        self.global_step = 0
        self.scaler = (
            torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None
        )

    def _setup_model(self):
        """Initialize the model."""
        self.model = C2DISRFactory.create_stage1_model(
            feature_dim=self.feature_dim,
            num_hiet_blocks=self.num_hiet_blocks,
            num_heads=self.num_heads,
        )
        self.model = self.model.to(self.device)

        # Print model info
        model_info = C2DISRFactory.get_model_info(self.model)
        print(f"Model initialized: {model_info}")

    def _setup_data(self):
        """Initialize dataset and dataloader."""
        self.dataset = ContinuousScaleData(
            hr_dir=self.data_path,
            min_scale=self.min_scale,
            max_scale=self.max_scale,
            patch_size=192,
            augment=True,
        )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )

        print(f"Dataset initialized: {len(self.dataset)} samples")

    def _setup_optimizer(self):
        """Initialize optimizer."""
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr_max,
            betas=(0.9, 0.999),
            weight_decay=0.0,
        )

    def _setup_scheduler(self):
        """Initialize learning rate scheduler."""
        self.scheduler = WarmupCosineScheduler(
            optimizer=self.optimizer,
            warmup_epochs=self.warmup_epochs,
            max_epochs=self.epochs,
            lr_max=self.lr_max,
            lr_min=self.lr_min,
        )

    def _setup_loss(self):
        """Initialize loss function."""
        self.criterion = L1Loss()

    def _setup_logging(self):
        """Initialize logging."""
        self.writer = SummaryWriter(log_dir=self.log_path)

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = []
        epoch_psnr = []
        epoch_ssim = []

        pbar = tqdm(self.dataloader, desc=f"Epoch {self.current_epoch}")

        for batch_idx, hr_imgs in enumerate(pbar):
            # Move to device
            hr_imgs = hr_imgs.to(self.device)

            # Sample scale for the batch
            scale = random.uniform(self.min_scale, self.max_scale)

            # Downsample hr_imgs to lr_imgs
            lr_h, lr_w = (
                int(self.dataset.patch_size / scale),
                int(self.dataset.patch_size / scale),
            )
            lr_imgs = kornia.geometry.transform.resize(
                hr_imgs, (lr_h, lr_w), interpolation="bicubic"
            )

            # Forward pass
            self.optimizer.zero_grad()

            # Use autocast for mixed precision training
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            with torch.amp.autocast(
                device_type=device_type, enabled=(self.scaler is not None)
            ):
                sr_imgs = self.model(lr_imgs, scale)

                # Ensure sr_imgs matches hr_imgs size (handle rounding issues in HIIFL)
                if sr_imgs.shape != hr_imgs.shape:
                    sr_imgs = F.interpolate(
                        sr_imgs,
                        size=hr_imgs.shape[2:],
                        mode="bilinear",
                        align_corners=False,
                    )

                # Calculate loss
                loss = self.criterion(sr_imgs, hr_imgs)

            # Backward pass with scaler
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            # Calculate metrics
            with torch.no_grad():
                psnr, ssim = batch_metrics(sr_imgs, hr_imgs)

            # Update statistics
            epoch_losses.append(loss.item())
            epoch_psnr.append(psnr)
            epoch_ssim.append(ssim)

            # Update progress bar
            pbar.set_postfix(
                {
                    "Loss": f"{loss.item():.6f}",
                    "PSNR": f"{psnr:.2f}",
                    "SSIM": f"{ssim:.4f}",
                    "LR": f"{self.optimizer.param_groups[0]['lr']:.6f}",
                }
            )

            # Log to tensorboard
            if self.global_step % 100 == 0:
                self.writer.add_scalar("Train/Loss", loss.item(), self.global_step)
                self.writer.add_scalar("Train/PSNR", psnr, self.global_step)
                self.writer.add_scalar("Train/SSIM", ssim, self.global_step)
                self.writer.add_scalar(
                    "Train/LR", self.optimizer.param_groups[0]["lr"], self.global_step
                )

            self.global_step += 1

        # Calculate epoch averages
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_psnr = sum(epoch_psnr) / len(epoch_psnr)
        avg_ssim = sum(epoch_ssim) / len(epoch_ssim)

        return {"loss": avg_loss, "psnr": avg_psnr, "ssim": avg_ssim}

    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        val_losses = []
        val_psnr = []
        val_ssim = []

        with torch.no_grad():
            # Sample a few batches for validation
            val_loader = DataLoader(
                self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=2
            )

            for hr_imgs in tqdm(val_loader, desc="Validation", leave=False):
                # Move to device
                hr_imgs = hr_imgs.to(self.device)

                # Sample scale for the batch
                scale = random.uniform(self.min_scale, self.max_scale)

                # Downsample hr_imgs to lr_imgs
                lr_h, lr_w = (
                    int(self.dataset.patch_size / scale),
                    int(self.dataset.patch_size / scale),
                )
                lr_imgs = kornia.geometry.transform.resize(
                    hr_imgs, (lr_h, lr_w), interpolation="bicubic"
                )

                # Forward pass
                device_type = "cuda" if torch.cuda.is_available() else "cpu"
                with torch.amp.autocast(
                    device_type=device_type, enabled=(self.scaler is not None)
                ):
                    sr_imgs = self.model(lr_imgs, scale)

                    # Ensure sr_imgs matches hr_imgs size
                    if sr_imgs.shape != hr_imgs.shape:
                        sr_imgs = F.interpolate(
                            sr_imgs,
                            size=hr_imgs.shape[2:],
                            mode="bilinear",
                            align_corners=False,
                        )

                    # Calculate loss and metrics
                    loss = self.criterion(sr_imgs, hr_imgs)

                psnr, ssim = batch_metrics(sr_imgs, hr_imgs)

                val_losses.append(loss.item())
                val_psnr.append(psnr)
                val_ssim.append(ssim)

                # Only validate on a few batches
                if len(val_losses) >= 50:
                    break

        avg_loss = sum(val_losses) / len(val_losses)
        avg_psnr = sum(val_psnr) / len(val_psnr)
        avg_ssim = sum(val_ssim) / len(val_ssim)

        return {"loss": avg_loss, "psnr": avg_psnr, "ssim": avg_ssim}

    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_psnr": self.best_psnr,
            "config": self.config,
        }

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        # Save latest checkpoint
        checkpoint_path = os.path.join(self.save_path, "stage1_latest.pth")
        tmp_path = checkpoint_path + ".tmp"
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, checkpoint_path)

        if is_best:
            best_path = os.path.join(self.save_path, "stage1_best.pth")
            best_tmp = best_path + ".tmp"
            torch.save(checkpoint, best_tmp)
            os.replace(best_tmp, best_path)

        print(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.current_epoch = checkpoint["epoch"] + 1
        self.global_step = checkpoint["global_step"]
        self.best_psnr = checkpoint["best_psnr"]

        if "epochs" in checkpoint.get("config", {}) and checkpoint["config"]["epochs"] != self.epochs:
            print(
                f"Warning: epochs mismatch — checkpoint has {checkpoint['config']['epochs']}, "
                f"config has {self.epochs}. LR curve may be incorrect."
            )

        print(f"Checkpoint loaded: {checkpoint_path}")
        print(
            f"Resuming from epoch {self.current_epoch}, best PSNR: {self.best_psnr:.2f}"
        )

    def train(self, resume_from: Optional[str] = None):
        """Main training loop."""
        if resume_from:
            if "_best.pth" in resume_from:
                resume_from = resume_from.replace("_best.pth", "_latest.pth")
                print(f"Substituted best checkpoint with latest: {resume_from}")
            self.load_checkpoint(resume_from)
            self.log_path = f"{self.log_path}_resume_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(self.log_path, exist_ok=True)
            self.writer = SummaryWriter(log_dir=self.log_path)

        print(f"Starting Stage 1 training for {self.epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.lr_max} -> {self.lr_min}")

        start_time = time.time()

        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch

            # Train one epoch
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Update learning rate
            self.writer.add_scalar("Epoch/Val_SSIM", val_metrics["ssim"], epoch)

            # Check if this is the best model
            is_best = val_metrics["psnr"] > self.best_psnr
            if is_best:
                self.best_psnr = val_metrics["psnr"]

            # Save checkpoint
            if epoch % 10 == 0 or is_best:
                self.save_checkpoint(is_best)

            # Update learning rate
            self.scheduler.step()

            # Print epoch summary
            elapsed_time = time.time() - start_time
            print(
                f"Epoch {epoch}/{self.epochs} - "
                f"Train Loss: {train_metrics['loss']:.6f}, "
                f"Train PSNR: {train_metrics['psnr']:.2f}, "
                f"Val PSNR: {val_metrics['psnr']:.2f}, "
                f"Best PSNR: {self.best_psnr:.2f}, "
                f"Time: {elapsed_time / 3600:.2f}h"
            )

        # Save final checkpoint
        self.save_checkpoint()

        print(f"Training completed! Best PSNR: {self.best_psnr:.2f}")
        self.writer.close()


def get_default_config() -> Dict:
    """Get default configuration for Stage 1 training."""
    return {
        # Training parameters
        "epochs": 700,
        "batch_size": 8,
        "lr_max": 4e-4,
        "lr_min": 1e-6,
        "warmup_epochs": 50,
        "min_scale": 1.0,
        "max_scale": 4.0,
        # Model parameters
        "feature_dim": 96,
        "num_hiet_blocks": 3,
        "num_heads": 8,
        # Paths
        "data_path": "data/div2k",
        "save_path": "checkpoints",
        "log_path": "logs/stage1",
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Stage 1: Continuous-scale Pre-training"
    )
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument(
        "--auto_resume",
        action="store_true",
        default=True,
        help="Automatically resume from latest checkpoint",
    )
    args = parser.parse_args()

    config = get_default_config()
    trainer = Stage1Trainer(config)

    resume_path = args.resume
    if not resume_path and args.auto_resume:
        latest_path = os.path.join(config["save_path"], "stage1_latest.pth")
        if os.path.exists(latest_path):
            resume_path = latest_path
            print(f"Auto-resuming from {resume_path}")

    trainer.train(resume_from=resume_path)
