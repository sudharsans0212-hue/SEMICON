#!/usr/bin/env python3
"""
KLA Hackathon 2026 — Standalone inference / evaluation script.
Usage:
    python inference.py --input_dir /path/to/NoisyLR --output_dir /path/to/restored --weights best_finetuned.pth
"""
import argparse
import os
import glob
import time

import numpy as np
import torch
import torch.nn as nn


# ----------------------------
# Model definition (self-contained — must match training exactly)
# ----------------------------
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
    """1-channel input (LR, noisy) -> 1-channel output at 2x resolution."""
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


# ----------------------------
# Inference pipeline
# ----------------------------
def load_model(weights_path, device):
    ckpt = torch.load(weights_path, map_location=device)
    cfg = ckpt.get('config', {'ch': 64, 'num_blocks': 8, 'growth': 32})
    model = KLARestorer(**cfg).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


def run_inference(input_dir, output_dir, weights_path, batch_size=8):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = load_model(weights_path, device)

    input_files = sorted(glob.glob(os.path.join(input_dir, '*.npy')))
    if not input_files:
        raise RuntimeError(f"No .npy files found in {input_dir}")
    print(f"Found {len(input_files)} input images.")

    total_t0 = time.time()
    n_done = 0

    with torch.no_grad():
        for i in range(0, len(input_files), batch_size):
            batch_files = input_files[i:i + batch_size]

            # ---- read + preprocess ----
            arrs = [np.load(f).astype(np.float32) for f in batch_files]
            batch = np.stack(arrs, axis=0)                      # (B,H,W)
            batch_t = torch.from_numpy(batch).unsqueeze(1)       # (B,1,H,W)
            batch_t = batch_t.to(device, non_blocking=True)

            # ---- inference ----
            with torch.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                pred = model(batch_t)
            pred = torch.clamp(pred, 0.0, 1.0)                   # KLA scores exactly what we save
            pred_np = pred.squeeze(1).float().cpu().numpy()      # (B,H,W)

            # ---- save ----
            for f, out_arr in zip(batch_files, pred_np):
                fname = os.path.basename(f)
                np.save(os.path.join(output_dir, fname), out_arr.astype(np.float32))
                n_done += 1

    total_time = time.time() - total_t0
    print(f"\nProcessed {n_done} images in {total_time:.2f}s "
          f"({total_time / n_done * 1000:.2f} ms/image, end-to-end incl. I/O)")
    print(f"Restored images saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="KLA restoration inference script")
    parser.add_argument('--input_dir', type=str, required=True, help="Directory of degraded .npy images")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to write restored .npy images")
    parser.add_argument('--weights', type=str, default='weights/best_finetuned.pth', help="Path to model checkpoint")
    parser.add_argument('--batch_size', type=int, default=8, help="Inference batch size")
    args = parser.parse_args()

    run_inference(args.input_dir, args.output_dir, args.weights, args.batch_size)


if __name__ == '__main__':
    main()
