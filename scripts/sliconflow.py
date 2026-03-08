"""Build CLIP-friendly motion text from style and movement metadata via Qwen3.5 API."""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

from openai import OpenAI
from tqdm.auto import tqdm

DEFAULT_INPUT = "data/Q1/100STYLE/NPZ_Style_Movement_Metadata.csv"
DEFAULT_OUTPUT = "tmp/NPZ_Style_Movement_Metadata_with_clip_text.csv"
DEFAULT_CACHE = "tmp/NPZ_Style_Movement_prompt_cache.csv"
DEFAULT_MODEL = "Pro/deepseek-ai/DeepSeek-V3.2"
DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"

# Keep backward compatibility with the previous script behavior.
DEFAULT_API_KEY = "sk-ftylzstkkpxtxvwribwvcbsadlfuadgayetxfuvlnbsrcikj"

SYSTEM_PROMPT = (
    "You are an expert in motion captioning for CLIP embedding. "
    "Given two independent descriptions from a motion capture dataset:\n"
    '- Upper body style: "{style_description}"\n'
    '- Lower body movement: "{movement_type}"\n'
    "Create one concise natural English sentence (<=30 words) describing full-body motion. "
    "Use clear action verbs, no redundancy, and no explanation. "
    'Start with "A person" or "Human motion of". '
    "Output only the final sentence."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CLIP text prompts from style_description + movement_type."
    )
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache", type=str, default=DEFAULT_CACHE)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-wait", type=float, default=1.5)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only process first N rows for quick testing. 0 means all rows.",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(
    path: Path, rows: list[dict[str, str]], fieldnames: list[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_key(style_description: str, movement_label: str) -> str:
    return f"{style_description}|||{movement_label}"


def load_cache(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    cache: dict[str, str] = {}
    rows = read_csv_rows(path)
    for row in rows:
        key = row.get("pair_key", "")
        value = row.get("clip_text_prompt", "")
        if key:
            cache[key] = value
    return cache


def save_cache(path: Path, cache: dict[str, str]) -> None:
    rows = [
        {"pair_key": key, "clip_text_prompt": value}
        for key, value in sorted(cache.items())
    ]
    write_csv_rows(path, rows, ["pair_key", "clip_text_prompt"])


def build_user_prompt(style_description: str, movement_label: str) -> str:
    return (
        f'style_description="{style_description}"\n' f'movement_type="{movement_label}"'
    )


def request_caption(
    client: OpenAI,
    model: str,
    style_description: str,
    movement_label: str,
    temperature: float,
    max_tokens: int,
    retries: int,
    retry_wait: float,
) -> str:
    user_prompt = build_user_prompt(style_description, movement_label)
    last_error = ""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )
            text = (response.choices[0].message.content or "").strip()
            if text:
                return " ".join(text.split())
            last_error = "empty response"
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
        if attempt < retries - 1:
            time.sleep(retry_wait * (attempt + 1))
    raise RuntimeError(f"caption request failed: {last_error}")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    cache_path = Path(args.cache).expanduser().resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    api_key = os.getenv("SILICONFLOW_API_KEY", DEFAULT_API_KEY)
    client = OpenAI(api_key=api_key, base_url=args.base_url)

    rows = read_csv_rows(input_path)
    if args.limit > 0:
        rows = rows[: args.limit]
    cache = load_cache(cache_path)

    output_rows: list[dict[str, str]] = []

    for row in tqdm(rows, desc="Generating CLIP prompts", unit="row"):
        style_description = (row.get("style_description") or "").strip()
        movement_label = (row.get("movement_type_label") or "").strip()
        pair_key = make_key(style_description, movement_label)

        prompt = cache.get(pair_key, "")
        if not prompt:
            prompt = request_caption(
                client=client,
                model=args.model,
                style_description=style_description,
                movement_label=movement_label,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                retries=args.retries,
                retry_wait=args.retry_wait,
            )
            cache[pair_key] = prompt
            save_cache(cache_path, cache)

        out = dict(row)
        out["clip_text_prompt"] = prompt
        out["llm_model"] = args.model
        output_rows.append(out)

    fieldnames = list(rows[0].keys()) + [
        "clip_text_prompt",
        "llm_model",
    ]
    write_csv_rows(output_path, output_rows, fieldnames)
    print(f"Done. output={output_path}")
    print(f"Done. cache={cache_path}")
    print(f"total_rows={len(output_rows)}")


if __name__ == "__main__":
    main()
