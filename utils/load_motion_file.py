"""Utilities to parse motion-file YAML declarations into NPZ path groups."""

from __future__ import annotations

import glob
import os
from typing import Dict, List

import yaml


def read_yaml_file(file_path: str) -> Dict:
    """Reads a YAML file and returns parsed content.

    Args:
        file_path: YAML file path.

    Returns:
        Parsed YAML dictionary. Returns empty dictionary on read/parse failure.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
        return data if data is not None else {}
    except (FileNotFoundError, yaml.YAMLError):
        return {}


def collect_npz_paths(yaml_path: str = "motion_file.yaml") -> Dict[str, List[str]]:
    """Collects grouped NPZ paths from motion YAML configuration.

    Args:
        yaml_path: Path to the YAML configuration file.

    Returns:
        Dictionary keyed by motion group name, each value is a sorted NPZ path
        list after include/exclude processing.
    """
    data = read_yaml_file(yaml_path)
    motion_groups = data.get("motion_group", {})
    result: Dict[str, List[str]] = {}

    for group_name, group_data in motion_groups.items():
        file_names = group_data.get("file_name", [])
        folder_names = group_data.get("folder_name", [])
        wo_file_names = group_data.get("wo_file_name", [])
        wo_folder_names = group_data.get("wo_folder_name", [])

        npz_paths: set[str] = set()
        basenames: set[str] = set()

        for path in file_names:
            if path.endswith(".npz") and os.path.exists(path):
                npz_paths.add(path)
                basenames.add(os.path.basename(path))

        for folder in folder_names:
            if os.path.isdir(folder):
                pattern = os.path.join(folder, "**", "*.npz")
                for npz_file in glob.glob(pattern, recursive=True):
                    basename = os.path.basename(npz_file)
                    if basename not in basenames:
                        npz_paths.add(npz_file)
                        basenames.add(basename)

        for wo_path in wo_file_names:
            npz_paths.discard(wo_path)

        for wo_folder in wo_folder_names:
            if os.path.isdir(wo_folder):
                pattern = os.path.join(wo_folder, "**", "*.npz")
                for npz_file in glob.glob(pattern, recursive=True):
                    npz_paths.discard(npz_file)

        result[group_name] = sorted(npz_paths)

    return result
