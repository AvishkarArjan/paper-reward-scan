import logging
import json
from pathlib import Path

from .schemas import SFTPair
from .utils import load_json, save_json

logger = logging.getLogger(__name__)


def compile_dataset(settings: dict) -> tuple[int, Path, Path]:
    output_dir = Path(settings["paths"]["output_dir"])
    pairs_dir = output_dir / "dataset" / "pairs"
    compiled_dir = output_dir / "dataset" / "compiled"

    pair_files = sorted(pairs_dir.glob("*.json"))
    if not pair_files:
        logger.warning("No SFT pair files found")
        return 0, None, None

    pairs = []
    for pf in pair_files:
        data = load_json(pf)
        if data:
            pairs.append(data)

    compiled_path = compiled_dir / "compiled.json"
    compiled_path_l = compiled_dir / "compiled.jsonl"

    save_json(compiled_path, pairs)

    with open(compiled_path_l, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    print()
    print(f"📦 Dataset Compiled")
    print(f"  SFT pairs: {len(pairs)}")
    print(f"  JSON (Alpaca format): {compiled_path}")
    print(f"  JSONL (Unsloth format): {compiled_path_l}")
    print()

    return len(pairs), compiled_path, compiled_path_l
