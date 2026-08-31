import typer
import logging
import contextlib
from pathlib import Path
from typing import Optional

from .utils import load_yaml, ensure_dir, timed
from .stats import begin_run, end_run, configure_stats, show_stats
from .ocr import ocr_papers
from .evaluator import evaluate_papers, print_evaluation_stats
from .extractor import extract_rewards
from .compiler import compile_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("tokenizers").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="prs",
    help="Paper Reward Scan — Build SFT datasets from UAV RL papers",
    no_args_is_help=True,
)

_FILE_LOGGING = {"attached": False}


def _enable_file_logging(output_dir: Path) -> None:
    """Mirror every log line to output/logs/prs.log with full-datetime stamps."""
    if _FILE_LOGGING["attached"]:
        return
    _FILE_LOGGING["attached"] = True
    log_path = output_dir / "logs" / "prs.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            "%Y-%m-%d %H:%M:%S",
        )
    )
    logging.getLogger().addHandler(fh)
    logger.info(f"── run ────────────────────────────────────────────────────")
    logger.info(f"log file: {log_path}")
    logger.info(f"───────────────────────────────────────────────────────────")


def _load_settings(enable_logging: bool = True) -> dict:
    settings_path = Path("configs/settings.yaml")
    if not settings_path.exists():
        logger.error("configs/settings.yaml not found")
        raise typer.Exit(1)
    settings = load_yaml(settings_path)
    output_dir = Path(settings["paths"]["output_dir"])
    configure_stats(output_dir / "logs" / "stats.json")
    if enable_logging:
        _enable_file_logging(output_dir)
    return settings


@contextlib.contextmanager
def _tracked(command: str, **meta):
    """Open a stats run record and always flush it (even on failure)."""
    begin_run(command, **meta)
    try:
        yield
    finally:
        end_run()


def _stats_path() -> Path:
    from .stats import get_stats_path
    path = get_stats_path()
    if path is None:
        settings = _load_settings(enable_logging=False)
        path = Path(settings["paths"]["output_dir"]) / "logs" / "stats.json"
    return path


@app.command()
def ocr(
    paper: Optional[str] = typer.Argument(
        None, help="Path to a single PDF, or directory (default: PAPERS/)"
    ),
    force: bool = typer.Option(
        False, "-f", "--force", help="Re-OCR already cached papers"
    ),
):
    """OCR papers into clean text + formulas (Unlimited-OCR)."""
    settings = _load_settings()

    paper_path = Path(paper) if paper else Path(settings["paths"]["papers_dir"])

    ensure_dir(Path(settings["paths"]["output_dir"]) / "ocr")

    with _tracked("ocr", force=force), timed("step: OCR", step=True):
        ocr_papers(paper_path, settings, force=force)


@app.command()
def evaluate(
    paper: Optional[str] = typer.Argument(
        None, help="Path to a single PDF, or directory (default: PAPERS/)"
    ),
    model: Optional[str] = typer.Option(
        None, "-m", "--model", help="Model name (HF or provider/name)"
    ),
    force: bool = typer.Option(
        False, "-f", "--force", help="Re-evaluate already cached papers"
    ),
    raw: bool = typer.Option(
        False, "-r", "--raw", help="Allow pypdf fallback when OCR text is missing"
    ),
):
    """Evaluate papers and shortlist those with quality reward functions."""
    settings = _load_settings()
    model_name = model or settings["model"]["default"]

    paper_path = Path(paper) if paper else Path(settings["paths"]["papers_dir"])

    ensure_dir(Path(settings["paths"]["output_dir"]) / "metadata")
    ensure_dir(Path(settings["paths"]["output_dir"]) / "evaluations")

    with _tracked("evaluate", model=model_name, force=force, fallback=raw), timed("step: evaluate", step=True):
        results, total, skipped = evaluate_papers(paper_path, settings, model_name, force=force, fallback=raw)
    if results:
        print_evaluation_stats(results, total, skipped)
    else:
        print("\nNo papers to evaluate.\n")


@app.command()
def extract(
    paper: Optional[str] = typer.Argument(
        None, help="Path to a single PDF, or directory (default: PAPERS/)"
    ),
    model: Optional[str] = typer.Option(
        None, "-m", "--model", help="Model name (HF or provider/name)"
    ),
    force: bool = typer.Option(
        False, "-f", "--force", help="Re-extract already cached papers"
    ),
    raw: bool = typer.Option(
        False, "-r", "--raw", help="Allow pypdf fallback when OCR text is missing"
    ),
):
    """Extract reward functions from accepted papers."""
    settings = _load_settings()
    model_name = model or settings["model"]["default"]

    paper_path = Path(paper) if paper else Path(settings["paths"]["papers_dir"])

    ensure_dir(Path(settings["paths"]["output_dir"]) / "extractions")
    ensure_dir(Path(settings["paths"]["output_dir"]) / "dataset" / "pairs")

    with _tracked("extract", model=model_name, force=force, fallback=raw), timed("step: extract", step=True):
        extract_rewards(paper_path, settings, model_name, force=force, fallback=raw)


@app.command()
def compile():
    """Compile individual SFT pairs into a single dataset."""
    settings = _load_settings()
    with _tracked("compile"), timed("step: compile", step=True):
        compile_dataset(settings)


@app.command()
def status():
    """Show pipeline status overview."""
    settings = _load_settings(enable_logging=False)
    output_dir = Path(settings["paths"]["output_dir"])
    papers_dir = Path(settings["paths"]["papers_dir"])

    total_papers = len(list(papers_dir.glob("*.pdf")))
    ocred = len(list((output_dir / "ocr").glob("*.md")))
    evaluateed = len(list((output_dir / "evaluations").glob("*.json")))
    extracted = len(list((output_dir / "extractions").glob("*.json")))
    pairs = len(list((output_dir / "dataset" / "pairs").glob("*.json")))
    compiled = (output_dir / "dataset" / "compiled" / "compiled.json").exists()

    accepted = 0
    for af in (output_dir / "evaluations").glob("*.json"):
        import json
        with open(af) as f:
            data = json.load(f)
            if data.get("passes_quality"):
                accepted += 1

    print()
    print("📊 Pipeline Status")
    print(f"  Papers in directory: {total_papers}")
    print(f"  OCR'd: {ocred}/{total_papers}")
    print(f"  Evaluateed: {evaluateed}/{total_papers}")
    if evaluateed:
        print(f"    Accepted: {accepted}")
        print(f"    Rejected: {evaluateed - accepted}")
    print(f"  Extracted: {extracted}/{accepted if accepted else '?'}")
    print(f"  SFT pairs: {pairs}")
    print(f"  Compiled: {'✅ yes' if compiled else '❌ no'}")
    print()


@app.command()
def run_all(
    model: Optional[str] = typer.Option(
        None, "-m", "--model", help="Model name (HF or provider/name)"
    ),
    force: bool = typer.Option(
        False, "-f", "--force", help="Re-process already cached papers"
    ),
    raw: bool = typer.Option(
        False, "-r", "--raw", help="Allow pypdf fallback when OCR text is missing"
    ),
):
    """Run full pipeline: ocr → evaluate → extract → compile."""
    settings = _load_settings()
    model_name = model or settings["model"]["default"]
    paper_path = Path(settings["paths"]["papers_dir"])

    typer.echo("=== Step 0: OCR ===")
    ensure_dir(Path(settings["paths"]["output_dir"]) / "ocr")

    with _tracked("run-all", model=model_name, force=force, fallback=raw), timed("step: OCR", step=True):
        ocr_papers(paper_path, settings, force=force)

    typer.echo("=== Step 1: Evaluate ===")
    ensure_dir(Path(settings["paths"]["output_dir"]) / "metadata")
    ensure_dir(Path(settings["paths"]["output_dir"]) / "evaluations")

    with timed("step: evaluate", step=True):
        results, total, skipped = evaluate_papers(paper_path, settings, model_name, force=force, fallback=raw)
    if results:
        print_evaluation_stats(results, total, skipped)

    typer.echo("=== Step 2: Extract ===")
    ensure_dir(Path(settings["paths"]["output_dir"]) / "extractions")
    ensure_dir(Path(settings["paths"]["output_dir"]) / "dataset" / "pairs")

    with timed("step: extract", step=True):
        extract_rewards(paper_path, settings, model_name, force=force, fallback=raw)

    typer.echo("=== Step 3: Compile ===")
    with timed("step: compile", step=True):
        compile_dataset(settings)

    typer.echo("✅ Pipeline complete!")


@app.command()
def stats(
    limit: int = typer.Option(
        0, "-n", "--limit", help="Show only the N most recent runs (0 = all)"
    ),
):
    """Show per-run / per-step / per-paper timing stats (rich tables)."""
    show_stats(_stats_path(), limit=limit)


def main():
    app()


if __name__ == "__main__":
    main()
