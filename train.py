#!/usr/bin/env python3
"""
KLA Hackathon 2026 — Training script.
Reproduces best_finetuned.pth from scratch: base training (Charbonnier + MS-SSIM),
then perceptual fine-tuning (adds VGG loss) to reduce over-smoothing.

Usage:
    python train.py --data_dir /path/to/train --output_dir weights/
"""
import argparse
import os
import glob
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pytorch_msssim import MS_SSIM
import torchvision.models as models


# ============================================================
# Dataset
# ============================================================
class KLADataset(Dataset):
    def __init__(self, gt_dir, lr_dir, train=True, crop_size=None):
        self.gt_files = sorted(glob.glob(os.path.join(gt_dir, '*.npy')))
        self.lr_files = sorted(glob.glob(os.path.join(lr_dir, '*.npy')))
        assert len(self.gt_files) == len(self.lr_files), "GT/NoisyLR count mismatch"
        self.train = train
        self.crop_size = crop_size

    def __len__(self):
        return len(self.gt_files)

    def __getitem__(self, idx):
        gt = np.load(self.gt_files[idx]).astype(np.float32)
        lr = np.load(self.lr_files[idx]).astype(np.float32)

        if self.train:
            if self.crop_size:
                h, w = lr.shape
                cs = self.crop_size
                top = random.randint(0, h - cs)
                left = random.randint(0, w - cs)
                lr = lr[top:top+cs, left:left+cs]
                gt = gt[top*2:(top+cs)*2, left*2:(left+cs)*2]

            if random.random() < 0.5:
                lr, gt = np.fliplr(lr).copy(), np.fliplr(gt).copy()
            if random.random() < 0.5:
                lr, gt = np.flipud(lr).copy(), np.flipud(gt).copy()
            k = random.randint(0, 3)
            if k:
                lr, gt = np.rot90(lr, k).copy(), np.rot90(gt, k).copy()

        lr_t = torch.from_numpy(lr).unsqueeze(0)
        gt_t = torch.from_numpy(gt).unsqueeze(0)
        return lr_t, gt_t, os.path.basename(self.gt_files[idx])


class KLASubset(KLADataset):
    def __init__(self, gt_dir, lr_dir, file_indices, train=True, crop_size=None):
        super().__init__(gt_dir, lr_dir, train, crop_size)
        self.gt_files = [self.gt_files[i] for i in file_indices]
        self.lr_files = [self.lr_files[i] for i in file_indices]


# ============================================================
# Model — RRDB-based 2x restorer (grayscale)
# ============================================================
class ResidualDenseBlock(nn.Module):
    def __init__(self, ch=64, growth=32):
        super().__init__()
        self.c1 = nn.Conv2d(ch, growth, 3, 1, 1)
        self.c2 = nn.Conv2d(ch + growth, growth, 3, 1, 1)
        self.c3 = nn.Conv2d(ch + 2 * growth, growth, 3, 1, 1)
        self.c4 = nn.Conv2d(ch + 3 * growth, growth, 3, 1, 1)
        self.c5 = nn.Conv2d(ch + 4 * growth, ch, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.act(self.c1(x))
        x2 = self.act(self.c2(torch.cat([x, x1], 1)))
        x3 = self.act(self.c3(torch.cat([x, x1, x2], 1)))
        x4 = self.act(self.c4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.c5(torch.cat([x, x1, x2, x3, x4], 1))
        return x + 0.2 * x5


class RRDB(nn.Module):
    def __init__(self, ch=64, growth=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(ch, growth)
        self.rdb2 = ResidualDenseBlock(ch, growth)
        self.rdb3 = ResidualDenseBlock(ch, growth)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return x + 0.2 * out


class KLARestorer(nn.Module):
    def __init__(self, ch=64, num_blocks=8, growth=32):
        super().__init__()
        self.head = nn.Conv2d(1, ch, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(ch, growth) for _ in range(num_blocks)])
        self.body_conv = nn.Conv2d(ch, ch, 3, 1, 1)
        self.up = nn.Sequential(
            nn.Conv2d(ch, ch * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.tail = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch, 1, 3, 1, 1),
        )

    def forward(self, x):
        feat = self.head(x)
        body_out = self.body_conv(self.body(feat))
        feat = feat + body_out
        feat = self.up(feat)
        return self.tail(feat)


# ============================================================
# Losses
# ============================================================
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps ** 2))


class CombinedLoss(nn.Module):
    """Stage 1 loss: Charbonnier + MS-SSIM.
    NOTE: Charbonnier uses raw (unclamped) pred/target deliberately — clamping
    would zero the gradient for any out-of-[0,1] prediction, stalling training.
    MS-SSIM clamps internally since it requires bounded inputs."""
    def __init__(self, alpha=0.84):
        super().__init__()
        self.charbonnier = CharbonnierLoss()
        self.msssim = MS_SSIM(data_range=1.0, channel=1)
        self.alpha = alpha

    def forward(self, pred, target):
        pred_c = torch.clamp(pred, 0, 1)
        target_c = torch.clamp(target, 0, 1)
        l_pix = self.charbonnier(pred, target)
        l_ssim = 1 - self.msssim(pred_c, target_c)
        return (1 - self.alpha) * l_pix + self.alpha * l_ssim


class VGGPerceptualLoss(nn.Module):
    def __init__(self, layer_idx=16):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:layer_idx].eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        pred3 = pred.repeat(1, 3, 1, 1)
        target3 = target.repeat(1, 3, 1, 1)
        pred_n = (pred3 - self.mean) / self.std
        target_n = (target3 - self.mean) / self.std
        return F.l1_loss(self.vgg(pred_n), self.vgg(target_n))


class CombinedLossV2(nn.Module):
    """Stage 2 loss: adds a VGG perceptual term to Stage 1's loss,
    to counter over-smoothing from pixel/structural losses alone."""
    def __init__(self, alpha=0.84, perceptual_weight=0.1):
        super().__init__()
        self.charbonnier = CharbonnierLoss()
        self.msssim = MS_SSIM(data_range=1.0, channel=1)
        self.perceptual = VGGPerceptualLoss()
        self.alpha = alpha
        self.perceptual_weight = perceptual_weight

    def forward(self, pred, target):
        pred_c = torch.clamp(pred, 0, 1)
        target_c = torch.clamp(target, 0, 1)
        l_pix = self.charbonnier(pred, target)
        l_ssim = 1 - self.msssim(pred_c, target_c)
        l_perc = self.perceptual(pred_c, target_c)
        base = (1 - self.alpha) * l_pix + self.alpha * l_ssim
        return base + self.perceptual_weight * l_perc


# ============================================================
# Training loop (shared by both stages)
# ============================================================
def train_stage(model, criterion, train_loader, val_loader, device, num_epochs,
                 lr, warmup_steps, ckpt_dir, best_filename, latest_filename,
                 resume_from=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
    total_steps = len(train_loader) * num_epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159265)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu')

    start_epoch = 0
    best_val_loss = float('inf')
    if resume_from and os.path.exists(resume_from):
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded starting weights from {resume_from}")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        nan_batches = 0

        for lr_img, gt_img, _ in train_loader:
            lr_img = lr_img.to(device, non_blocking=True)
            gt_img = gt_img.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type):
                pred = model(lr_img)
            # Loss computed in fp32 — MS-SSIM/Charbonnier are numerically fragile in fp16
            loss = criterion(pred.float(), gt_img.float())

            if not torch.isfinite(loss):
                nan_batches += 1
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += loss.item()

        train_loss = running_loss / max(1, len(train_loader) - nan_batches)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for lr_img, gt_img, _ in val_loader:
                lr_img, gt_img = lr_img.to(device), gt_img.to(device)
                with torch.amp.autocast(device.type):
                    pred = model(lr_img)
                val_loss += criterion(pred.float(), gt_img.float()).item()
        val_loss /= len(val_loader)

        dt = time.time() - t0
        print(f"Epoch {epoch+1}/{num_epochs}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}  "
              f"nan_batches={nan_batches}  time={dt:.1f}s")

        ckpt_dict = {
            'epoch': epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 'val_loss': val_loss,
            'config': {'ch': 64, 'num_blocks': 8, 'growth': 32},
        }
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt_dict, os.path.join(ckpt_dir, best_filename))
        torch.save(ckpt_dict, os.path.join(ckpt_dir, latest_filename))

    return best_val_loss


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="KLA restoration training script")
    parser.add_argument('--data_dir', type=str, default='data/train',
                         help="Directory containing GT/ and NoisyLR/ subfolders")
    parser.add_argument('--output_dir', type=str, default='weights',
                         help="Directory to save checkpoints")
    parser.add_argument('--base_epochs', type=int, default=100)
    parser.add_argument('--finetune_epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--val_frac', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    gt_dir = os.path.join(args.data_dir, 'GT')
    lr_dir = os.path.join(args.data_dir, 'NoisyLR')

    all_gt = sorted(glob.glob(os.path.join(gt_dir, '*.npy')))
    idxs = list(range(len(all_gt)))
    random.seed(args.seed)
    random.shuffle(idxs)
    n_val = int(len(idxs) * args.val_frac)
    val_idxs, train_idxs = sorted(idxs[:n_val]), sorted(idxs[n_val:])

    train_ds = KLASubset(gt_dir, lr_dir, train_idxs, train=True, crop_size=96)
    val_ds = KLASubset(gt_dir, lr_dir, val_idxs, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)
    print(f"Train pairs: {len(train_ds)}  Val pairs: {len(val_ds)}")

    model = KLARestorer(ch=64, num_blocks=8).to(device)

    # ---- Stage 1: base training (Charbonnier + MS-SSIM) ----
    print("\n=== Stage 1: base training ===")
    criterion_v1 = CombinedLoss(alpha=0.84).to(device)
    train_stage(model, criterion_v1, train_loader, val_loader, device,
                num_epochs=args.base_epochs, lr=1e-4, warmup_steps=500,
                ckpt_dir=args.output_dir, best_filename='best.pth', latest_filename='latest.pth')

    # ---- Stage 2: perceptual fine-tune ----
    print("\n=== Stage 2: perceptual fine-tune ===")
    criterion_v2 = CombinedLossV2(alpha=0.84, perceptual_weight=0.1).to(device)
    train_stage(model, criterion_v2, train_loader, val_loader, device,
                num_epochs=args.finetune_epochs, lr=2e-5, warmup_steps=0,
                ckpt_dir=args.output_dir, best_filename='best_finetuned.pth',
                latest_filename='latest_finetuned.pth',
                resume_from=os.path.join(args.output_dir, 'best.pth'))

    print(f"\nDone. Final checkpoint: {os.path.join(args.output_dir, 'best_finetuned.pth')}")


if __name__ == '__main__':
    main()
