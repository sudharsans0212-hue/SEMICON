# KLA Hackathon 2026 — AI-Based Restoration of Degraded Images

Restoration pipeline for semiconductor inspection images degraded by speckle noise, additive Gaussian noise, and 2x downsampling. Restores 128x128 grayscale NoisyLR inputs to 256x256 clean outputs.

## Repository Structure

- inference.py — standalone evaluation script (run this)
- train.py — training script (reproduces best_finetuned.pth from scratch)
- requirements.txt — exact pip freeze from training environment
- weights/best_finetuned.pth — final trained model checkpoint
- results/metrics.md — PSNR / SSIM / LPIPS results, baseline comparison
- results/restored_test/ — model outputs on the official hidden test set

## Environment Setup

Run: pip install -r requirements.txt

Tested on Python 3.12, PyTorch with CUDA support, single NVIDIA GPU (trained on T4, compatible with H100 for inference).

## Running Inference (Evaluation)

python inference.py --input_dir /path/to/NoisyLR --output_dir /path/to/restored_output --weights weights/best_finetuned.pth

Arguments:
- --input_dir (required): directory containing degraded .npy input images
- --output_dir (required): directory to write restored .npy output images (created if it doesn't exist)
- --weights (optional, default weights/best_finetuned.pth): path to model checkpoint
- --batch_size (optional, default 8): inference batch size

Behavior:
- Loads all .npy files from input_dir
- Runs restoration (denoising + 2x super-resolution) on GPU (falls back to CPU if unavailable)
- Clips output to [0,1] range before saving
- Saves each restored image to output_dir with the same filename as its input
- Prints total images processed and end-to-end ms/image (includes disk I/O, preprocessing, GPU transfer, inference, postprocessing, and saving)

No manual edits to the script are required — all paths are passed via command-line arguments.

## Reproducing Training

Run: python train.py

train.py reproduces the full training process from scratch:
1. Trains the base restoration model for 100 epochs (Charbonnier + MS-SSIM loss)
2. Fine-tunes for 15 additional epochs with an added VGG perceptual loss term to recover fine texture detail, saving the final checkpoint to weights/best_finetuned.pth

Expects the official KLA dataset at data/train/GT/ and data/train/NoisyLR/ (paired .npy files, matched by filename) — edit the DATA_DIR constant near the top of train.py if your local paths differ.

## Model

Custom RRDB-based (Residual-in-Residual Dense Block) restoration network — a compact ESRGAN-style generator adapted for single-channel (grayscale) input, with a 2x pixel-shuffle upsampling head. 8 RRDB blocks, 64 base channels.

## Results

See results/metrics.md for full PSNR / SSIM / LPIPS results and baseline comparison.

## Hardware

- Training: Google Colab, NVIDIA T4 GPU
- Mixed precision (AMP) training, gradient clipping, cosine LR schedule with warmup

## External Resources

- MS-SSIM: pytorch-msssim package
- LPIPS: lpips package (AlexNet backbone, ImageNet-pretrained weights)
- VGG16 perceptual loss: torchvision (ImageNet-pretrained weights)
