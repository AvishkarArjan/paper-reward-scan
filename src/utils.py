import time
import json
import yaml
import hashlib
import logging
import contextlib
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def hms(seconds: float) -> str:
    """Format a duration in seconds as '1h 2m 3s' (compact for short times)."""
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m {secs}s"


@contextlib.contextmanager
def timed(label: str, step: bool = False, stem: str = ""):
    """Log a start timestamp and completion duration for a labelled block.

    Logs a "started at" line before yielding, then a "done in" line afterwards,
    or a "FAILED after" line if the block raises. Records the timing in the
    global stats recorder too: pass `step=True` for a pipeline stage and
    `stem="<paper>"` for per-paper work.
    """
    from .stats import recorder

    started = time.monotonic()
    logger.info(f"{label} — started at {datetime.now():%Y-%m-%d %H:%M:%S}")
    if step:
        recorder.start_step(label)
    elif stem:
        recorder.start_paper(stem)
    try:
        yield
    except Exception:
        logger.error(f"{label} — FAILED after {hms(time.monotonic() - started)}")
        if step:
            recorder.end_step()
        elif stem:
            recorder.end_paper("failed")
        raise
    if step:
        recorder.end_step()
    elif stem:
        recorder.end_paper()
    logger.info(f"{label} — done in {hms(time.monotonic() - started)}")


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


def get_paper_text(pdf_file: Path, settings: dict, fallback: bool = True) -> str:
    """Paper text for the LLM stages: reads Unlimited-OCR output (text +
    formulas as LaTeX) from ``output/ocr/<stem>.md``.

    When *fallback* is ``True`` (the old default), raw pypdf extraction is
    used as a last resort.  When ``False`` (new default for pipeline commands),
    a missing OCR file is treated as an error and returns ``""`` so the caller
    can skip the paper with a clear message."""
    ocr_md = Path(settings["paths"]["output_dir"]) / "ocr" / f"{pdf_file.stem}.md"
    if ocr_md.exists():
        text = ocr_md.read_text(errors="ignore").strip()
        if text:
            return text
    if fallback:
        logger.warning(
            f"[{pdf_file.stem}] no OCR text found — using raw pypdf text (run `prs ocr`)"
        )
        return read_pdf(pdf_file)
    logger.error(
        f"[{pdf_file.stem}] no OCR text at {ocr_md} — "
        "skipping (use --raw to force pypdf fallback)"
    )
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
