#!/usr/bin/env bash
set -euo pipefail

# Runs the smallest independent unit test suite required for core modules.
python -mpytest -q tests/test_vae_models.py tests/test_training_minimal.py tests/test_conv_vq_fsq.py tests/test_batch_protocol_sequence.py tests/test_registry_minimal.py tests/test_hydra_config_minimal.py tests/test_motion_mimic_dataset.py tests/test_motion_yaml_loader_integration.py
