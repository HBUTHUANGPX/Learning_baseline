"""Encode clip_text_prompt with CLIP and save pooler_output by npz_file_name."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Iterator

import torch
from tqdm.auto import tqdm
try:
    from transformers import CLIPTextModel, CLIPTokenizer
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Missing dependency 'transformers'. "
        "Install it in your active env, e.g.:\n"
        "  pip install transformers"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build CLIP text pooler_output tokens from CSV prompts."
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default="tmp/NPZ_Style_Movement_Metadata_with_clip_text.csv",
    )
    parser.add_argument(
        "--clip-path",
        type=str,
        default="ckpts/clip-vit-large-patch14",
    )
    parser.add_argument(
        "--output-pt",
        type=str,
        default="tmp/clip_text_pooler_by_npz.pt",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-length", type=int, default=77)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def resolve_device(raw: str) -> torch.device:
    if raw == "cpu":
        return torch.device("cpu")
    if raw == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows


def batch_iter(items: list[dict[str, str]], batch_size: int) -> Iterator[list[dict[str, str]]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


@torch.no_grad()
def encode_prompts(
    rows: list[dict[str, str]],
    tokenizer: CLIPTokenizer,
    model: CLIPTextModel,
    device: torch.device,
    max_length: int,
    batch_size: int,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    progress = tqdm(total=len(rows), desc="Encoding CLIP prompts", unit="row")

    for batch_rows in batch_iter(rows, batch_size):
        prompts = [(row.get("clip_text_prompt") or "").strip() for row in batch_rows]
        names = [(row.get("npz_file_name") or "").strip() for row in batch_rows]

        enc = tokenizer(
            prompts,
            return_length=False,
            return_overflowing_tokens=False,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_attention_mask=True,
            return_tensors="pt",
        )
        out = model(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
        )
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            pooled = out.pooler_output
        else:
            # Fallback mean pooling with attention mask.
            attn = enc["attention_mask"].to(device).unsqueeze(-1).float()
            pooled = (out.last_hidden_state * attn).sum(dim=1) / attn.sum(dim=1).clamp(min=1e-6)
        pooled = pooled.detach().cpu()

        for idx, name in enumerate(names):
            if not name:
                continue
            result[name] = {
                "clip_text_prompt": prompts[idx],
                "token": pooled[idx],
                "pooler_output": pooled[idx],
            }
        progress.update(len(batch_rows))

    progress.close()
    return result


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0.")

    input_csv = Path(args.input_csv).expanduser().resolve()
    clip_path = Path(args.clip_path).expanduser().resolve()
    output_pt = Path(args.output_pt).expanduser().resolve()

    if not input_csv.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    if not clip_path.exists():
        raise FileNotFoundError(f"CLIP model path not found: {clip_path}")

    rows = read_rows(input_csv)
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("No rows found in input CSV.")

    required_cols = {"npz_file_name", "clip_text_prompt"}
    missing_cols = required_cols - set(rows[0].keys())
    if missing_cols:
        raise KeyError(f"Input CSV missing required columns: {sorted(missing_cols)}")

    device = resolve_device(args.device)
    tokenizer = CLIPTokenizer.from_pretrained(str(clip_path), max_length=args.max_length)
    model = CLIPTextModel.from_pretrained(str(clip_path)).to(device).eval()

    data = encode_prompts(
        rows=rows,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    output_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, output_pt)
    print(f"saved={output_pt}")
    print(f"num_items={len(data)}")
    print(f"device={device}")


if __name__ == "__main__":
    main()
