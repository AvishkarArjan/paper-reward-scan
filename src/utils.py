import json
import yaml
import hashlib
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_json(path: Path, data: Any, indent: int = 2) -> None:
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent, default=str)


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def read_pdf(path: Path) -> str:
    try:
        import pypdf
        reader = pypdf.PdfReader(str(path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text.strip()
    except Exception as e:
        logger.warning(f"Failed to read PDF {path}: {e}")
        return ""


def get_paper_files(directory: Path) -> list[Path]:
    if directory.is_file():
        if directory.suffix.lower() == ".pdf":
            return [directory]
        else:
            logger.warning(f"Not a PDF file: {directory}")
            return []
    if not directory.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return []
    return sorted(directory.glob("*.pdf"))


def extract_json_from_response(text: str) -> dict[str, Any] | None:
    text = text.strip()

    import re
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if json_match:
        candidate = json_match.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        candidate = text[brace_start : brace_end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    return None


def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[TRUNCATED]"


CONTENT_REGISTRY = ".content_registry.json"


def compute_file_hash(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def load_content_registry(output_dir: Path) -> dict[str, str]:
    path = output_dir / CONTENT_REGISTRY
    data = load_json(path)
    return data if data else {}


def save_content_registry(output_dir: Path, registry: dict[str, str]) -> None:
    path = output_dir / CONTENT_REGISTRY
    save_json(path, registry)
