# VAE Series Baseline

This repository provides a modular Variational Autoencoder (VAE) baseline with:

- `VanillaVAE`
- `BetaVAE`
- `ConvVAE`
- `VQVAE`
- `FSQVAE`

It also includes a protocol-based batch/observation pipeline for future
sequence and generative algorithm extensions.

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

Hydra layered config entrypoint:

```bash
python scripts/train_hydra.py
```

Hydra composition and override examples:

```bash
python scripts/train_hydra.py model=fsq data=mnist train.epochs=200
python scripts/train_hydra.py model=conv data=mnist optim.lr=5e-4 train.device=cpu
python scripts/train_hydra.py data=random_sequence data.sequence_length=128 data.sequence_feature_dim=32
```

Config layout:

- `configs/data/*.yaml`
- `configs/model/*.yaml`
- `configs/algo/*.yaml`
- `configs/optim/*.yaml`
- `configs/train/*.yaml`
- `configs/log/*.yaml`

Batch protocol with sequence dataset (pipeline demo):

```bash
python -m pytest -q tests/test_batch_protocol_sequence.py
```

Note:

- `fsq` now supports sequence reconstruction with a temporal Conv1d + FSQ path
  when using `data=random_sequence` or `data=motion_mimic` with
  `data.motion_as_sequence=true`.
- `conv`/`vq` remain image-model paths and expect 4D image batches.

Image models (`conv`, `vq`, `fsq`) example:

```bash
python scripts/train_vae.py --model conv --dataset random_binary --input-dim 784 --epochs 2
python scripts/train_vae.py --model vq --dataset random_binary --latent-dim 16 --num-embeddings 128 --epochs 2
python scripts/train_vae.py --model fsq --dataset random_binary --latent-dim 16 --fsq-levels 8 --epochs 2
```

Reconstruction loss mode:

- `--recon-loss-mode auto` (default): use BCE for `[0,1]` targets, otherwise MSE.
- `--recon-loss-mode bce`: force BCE (for normalized Bernoulli-style data).
- `--recon-loss-mode mse`: force MSE (recommended for continuous motion features).

Example for motion-like continuous data:

```bash
python scripts/train_hydra.py data=motion_mimic data.motion_group=xsens_bvh model=vanilla model.recon_loss_mode=mse
```

FSQ sequence reconstruction examples:

```bash
python scripts/train_hydra.py model=fsq data=random_sequence model.recon_loss_mode=mse
python scripts/train_hydra.py model=fsq data=motion_mimic data.motion_group=xsens_bvh data.motion_as_sequence=true model.recon_loss_mode=mse
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
