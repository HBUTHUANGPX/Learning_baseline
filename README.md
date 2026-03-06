# Motion VQ/FSQ Baseline

This repository is intentionally slimmed down to only keep:

- frame-level `VQVAE`
- frame-level `FSQVAE`
- motion NPZ loading and training pipeline

## Directory Layout

- `configs/`: Hydra configs (motion data + vq/fsq model + train/optim/log)
- `modules/`: motion data module, quantizers, frame VQ/FSQ models
- `scripts/`: training and evaluation scripts
- `utils/`: seed helper, tensorboard logger, motion-file YAML parser
- `tests/`: minimal pytest coverage

## Context-Conditioned Task

The training task is fixed as:

- encoder input: `n` history + current + `m` future
- decoder input: quantized latent + `n` history
- reconstruction target: current + `m` future

This corresponds to:

- `history_frames = n`
- `future_frames = m`

## Training

Hydra entrypoint:

```bash
python scripts/train_hydra.py \
  model=fsq \
  data.motion_group=xsens_bvh \
  data.history_frames=4 \
  data.future_frames=2 \
  train.epochs=200
```

CLI entrypoint:

```bash
python scripts/train_motion_vqvae.py \
  --model fsq \
  --motion-file-yaml configs/data/motion_file.yaml \
  --motion-group xsens_bvh \
  --history-frames 4 \
  --future-frames 2 \
  --epochs 200 \
  --batch-size 1024
```

## TensorBoard

Event files are saved at:

```text
log/<timestamp>/tensorboard
```

Start TensorBoard:

```bash
tensorboard --logdir ./log --port 6006
```

## MuJoCo Evaluation

```bash
python scripts/eval_mujoco_motion.py \
  --ckpt log/<timestamp>/checkpoint/epoch_200.pt \
  --mjcf /path/to/robot.xml \
  --motion-file-yaml configs/data/motion_file.yaml \
  --motion-group xsens_bvh \
  --motion-feature-keys joint_pos,joint_vel \
  --qpos-key joint_pos \
  --qpos-start-idx 0 \
  --qpos-slice-start 0
```

## Tests

```bash
python -m pytest -q
```
