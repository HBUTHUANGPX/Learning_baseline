"""Build combined metadata for 100STYLE NPZ files.

Outputs one CSV row per NPZ with:
- NPZ file path/name
- Style Name and Description from Dataset_List.csv
- MovementType code and readable label
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

MOVEMENT_TYPE_LABELS = {
    "BR": "Backwards Running",
    "BW": "Backwards Walking",
    "FR": "Forwards Running",
    "FW": "Forwards Walking",
    "ID": "Idling",
    "SR": "Sidestep Running",
    "SW": "Sidestep Walking",
    "TR1": "Transition 1",
    "TR2": "Transition 2",
    "TR3": "Transition 3",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build NPZ + style + movement metadata CSV."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="data/Q1/100STYLE",
        help="Root folder that contains style subfolders and Dataset_List.csv.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="data/Q1/100STYLE/Dataset_List.csv",
        help="Dataset style list csv path.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/Q1/100STYLE/NPZ_Style_Movement_Metadata.csv",
        help="Output combined metadata csv.",
    )
    return parser.parse_args()


def _load_style_description(csv_path: Path) -> dict[str, str]:
    style_to_desc: dict[str, str] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            style = (row.get("Style Name") or "").strip()
            desc = (row.get("Description") or "").strip()
            if style:
                style_to_desc[style] = desc
    return style_to_desc


def _extract_movement_code(npz_name: str) -> str:
    stem = Path(npz_name).stem
    if "_" not in stem:
        return ""
    return stem.rsplit("_", maxsplit=1)[-1]


def main() -> None:
    args = _parse_args()
    root = Path(args.root).expanduser().resolve()
    csv_path = Path(args.csv).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not root.is_dir():
        raise FileNotFoundError(f"Root directory not found: {root}")
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset list csv not found: {csv_path}")

    style_to_desc = _load_style_description(csv_path)
    npz_paths = sorted(root.glob("*/*.npz"))

    rows: list[dict[str, str]] = []
    for npz_path in npz_paths:
        style_name = npz_path.parent.name
        npz_file_name = npz_path.name
        movement_type = _extract_movement_code(npz_file_name)
        movement_label = MOVEMENT_TYPE_LABELS.get(movement_type, "Unknown")

        # Consistency check between folder style and filename prefix.
        filename_style_prefix = Path(npz_file_name).stem.rsplit("_", maxsplit=1)[0]
        style_match = str(filename_style_prefix == style_name)

        rows.append(
            {
                "npz_path": str(npz_path.relative_to(root.parent.parent.parent)),
                "npz_file_name": npz_file_name,
                "style_name": style_name,
                "style_description": style_to_desc.get(style_name, ""),
                "movement_type": movement_type,
                "movement_type_label": movement_label,
                "filename_style_prefix": filename_style_prefix,
                "style_name_matches_filename": style_match,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "npz_path",
        "npz_file_name",
        "style_name",
        "style_description",
        "movement_type",
        "movement_type_label",
        "filename_style_prefix",
        "style_name_matches_filename",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    missing_desc = sum(1 for row in rows if not row["style_description"])
    unknown_move = sum(1 for row in rows if row["movement_type_label"] == "Unknown")
    mismatch = sum(1 for row in rows if row["style_name_matches_filename"] == "False")
    print(f"rows={len(rows)}")
    print(f"missing_description={missing_desc}")
    print(f"unknown_movement_type={unknown_move}")
    print(f"style_filename_mismatch={mismatch}")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
