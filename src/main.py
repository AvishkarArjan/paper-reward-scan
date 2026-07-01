import typer
import logging
from pathlib import Path
from typing import Optional

from .utils import load_yaml, ensure_dir
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


def _load_settings() -> dict:
    settings_path = Path("configs/settings.yaml")
    if not settings_path.exists():
        logger.error("configs/settings.yaml not found")
        raise typer.Exit(1)
    return load_yaml(settings_path)


@app.command()
def evaluate(
    paper: Optional[str] = typer.Argument(
        None, help="Path to a single PDF, or directory (default: papers/)"
    ),
    model: Optional[str] = typer.Option(
        None, "-m", "--model", help="Model name (HF or provider/name)"
    ),
    force: bool = typer.Option(
        False, "-f", "--force", help="Re-evaluate already cached papers"
    ),
):
    """Evaluate papers and shortlist those with quality reward functions."""
    settings = _load_settings()
    model_name = model or settings["model"]["default"]

    paper_path = Path(paper) if paper else Path(settings["paths"]["papers_dir"])

    ensure_dir(Path(settings["paths"]["output_dir"]) / "metadata")
    ensure_dir(Path(settings["paths"]["output_dir"]) / "evaluations")

    results, total, skipped = evaluate_papers(paper_path, settings, model_name, force=force)
    if results:
        print_evaluation_stats(results, total, skipped)
    else:
        print("\nNo papers to evaluate.\n")


@app.command()
def extract(
    paper: Optional[str] = typer.Argument(
        None, help="Path to a single PDF, or directory (default: papers/)"
    ),
    model: Optional[str] = typer.Option(
        None, "-m", "--model", help="Model name (HF or provider/name)"
    ),
    force: bool = typer.Option(
        False, "-f", "--force", help="Re-extract already cached papers"
    ),
):
    """Extract reward functions from accepted papers."""
    settings = _load_settings()
    model_name = model or settings["model"]["default"]

    paper_path = Path(paper) if paper else Path(settings["paths"]["papers_dir"])

    ensure_dir(Path(settings["paths"]["output_dir"]) / "extractions")
    ensure_dir(Path(settings["paths"]["output_dir"]) / "dataset" / "pairs")

    extract_rewards(paper_path, settings, model_name, force=force)


@app.command()
def compile():
    """Compile individual SFT pairs into a single dataset."""
    settings = _load_settings()
    compile_dataset(settings)


@app.command()
def status():
    """Show pipeline status overview."""
    settings = _load_settings()
    output_dir = Path(settings["paths"]["output_dir"])
    papers_dir = Path(settings["paths"]["papers_dir"])

    total_papers = len(list(papers_dir.glob("*.pdf")))
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
):
    """Run full pipeline: evaluate → extract → compile."""
    settings = _load_settings()
    model_name = model or settings["model"]["default"]
    paper_path = Path(settings["paths"]["papers_dir"])

    typer.echo("=== Step 1: Evaluate ===")
    ensure_dir(Path(settings["paths"]["output_dir"]) / "metadata")
    ensure_dir(Path(settings["paths"]["output_dir"]) / "evaluations")

    results, total, skipped = evaluate_papers(paper_path, settings, model_name, force=force)
    if results:
        print_evaluation_stats(results, total, skipped)

    typer.echo("=== Step 2: Extract ===")
    ensure_dir(Path(settings["paths"]["output_dir"]) / "extractions")
    ensure_dir(Path(settings["paths"]["output_dir"]) / "dataset" / "pairs")

    extract_rewards(paper_path, settings, model_name, force=force)

    typer.echo("=== Step 3: Compile ===")
    compile_dataset(settings)

    typer.echo("✅ Pipeline complete!")


def main():
    app()


if __name__ == "__main__":
    main()
