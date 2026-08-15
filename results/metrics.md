# Validation Results

Evaluated on a held-out validation split (10% of training pairs, fixed by seed, never used in training or model selection).

| Method             | PSNR (dB) | SSIM   | LPIPS  |
|---------------------|-----------|--------|--------|
| Bicubic baseline    | 23.279    | 0.5554 | 0.4312 |
| Ours (final model)  | 28.001    | 0.7589 | 0.2280 |

## Notes

- Final model is a 2-stage trained RRDB restorer: 100 epochs base training (Charbonnier + MS-SSIM loss), followed by 15 epochs of fine-tuning with an added VGG perceptual loss term (weight 0.1) to reduce over-smoothing and recover fine texture detail.
- The perceptual fine-tune trades a small amount of pixel/structural fidelity (-0.13 dB PSNR, -0.011 SSIM) for a meaningful LPIPS improvement (-0.038, ~14% relative), visible in sharper restoration of fine structures (e.g. road markings, fibrous/thorny textures) compared to the base model.
- Bicubic baseline: standard bicubic upsampling of the NoisyLR input to GT resolution, no denoising, no learned component — sanity baseline confirming the model learns genuine restoration rather than trivial upsampling.
