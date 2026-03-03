#!/usr/bin/env bash
set -euo pipefail

# Runs the smallest independent unit test suite required for core modules.
python -mpytest -q tests/test_vae_models.py tests/test_training_minimal.py tests/test_conv_vq_fsq.py
