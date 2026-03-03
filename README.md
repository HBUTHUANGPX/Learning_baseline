# VAE Series Baseline

This repository provides a modular Variational Autoencoder (VAE) baseline with:

- `VanillaVAE`
- `BetaVAE`
- `ConvVAE`
- `VQVAE`
- `FSQVAE`

## Directory Layout

- `./data`: datasets
- `./log`: experiment logs (timestamp-based run folder)
  - `tensorboard`: TensorBoard event files
  - `checkpoint`: model checkpoints
  - `reconstructions`: reconstruction artifacts
- `./modules`: reusable neural modules
- `./utils`: logging and utility helpers
- `./scripts`: train script and minimal test script
- `./tests`: pytest unit tests

## Quick Start

```bash
python scripts/train_vae.py --model vanilla --dataset random_binary --epochs 2
```

Image models (`conv`, `vq`, `fsq`) example:

```bash
python scripts/train_vae.py --model conv --dataset random_binary --input-dim 784 --epochs 2
python scripts/train_vae.py --model vq --dataset random_binary --latent-dim 16 --num-embeddings 128 --epochs 2
python scripts/train_vae.py --model fsq --dataset random_binary --latent-dim 16 --fsq-levels 8 --epochs 2
```

For stronger reproducibility across runs:

```bash
python scripts/train_vae.py --model fsq --dataset mnist --epochs 200 --deterministic --device cpu
```

## TensorBoard

After training starts, event files are written to:

```text
./log/<timestamp>/tensorboard
```

Start TensorBoard:

```bash
tensorboard --logdir ./log --port 6006
```

Open in browser:

```text
http://127.0.0.1:6006
```

What to check:

- Scalars: `train/loss`, `train/recon_loss`, `train/kl_loss` or `train/quant_loss`
- Scalars: `val/*` metrics
- Reconstruction:
  - MLP models: histogram logging
  - Conv/VQ/FSQ models: image logging

## Minimal Tests

```bash
bash scripts/run_minimal_tests.sh
```
