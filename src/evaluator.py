import logging
from pathlib import Path
from typing import Optional

from .schemas import PaperMetadata, EvaluationResult
from .utils import (
    load_yaml, save_json, load_json, read_pdf,
    get_paper_files, truncate_text,
)
from .models import create_client, BaseClient

logger = logging.getLogger(__name__)


def evaluate_papers(
    paper_path: Path,
    settings: dict,
    model_name: str,
    force: bool = False,
) -> tuple[list[EvaluationResult], int, int]:
    papers_dir = Path(settings["paths"]["papers_dir"])
    output_dir = Path(settings["paths"]["output_dir"])
    metadata_dir = output_dir / "metadata"
    evaluations_dir = output_dir / "evaluations"

    threshold = settings["evaluation"]["quality_threshold"]
    max_chars = settings["evaluation"]["max_paper_chars"]
    temperature = settings["evaluation"]["temperature"]
    max_retries = settings["evaluation"]["max_retries"]

    papers = get_paper_files(paper_path)
    if not papers:
        logger.warning(f"No PDF files found at {paper_path}")
        return [], 0, 0

    prompt_data = load_yaml(Path("configs/prompts/evaluate.yaml"))
    system_prompt = prompt_data["system"].replace("THRESHOLD", str(threshold))
    user_template = prompt_data["user"]

    client = create_client(model_name, settings["model"]["hf_cache_dir"], settings.get("rate_limits"))
    results = []
    skipped = 0

    for pdf_file in papers:
        stem = pdf_file.stem
        metadata_file = metadata_dir / f"{stem}.json"
        evaluation_file = evaluations_dir / f"{stem}.json"

        if evaluation_file.exists() and not force:
            existing = load_json(evaluation_file)
            if existing:
                results.append(EvaluationResult(**existing))
                skipped += 1
                logger.info(f"[{stem}] → cached")
                continue

        pdf_text = read_pdf(pdf_file)
        if not pdf_text:
            logger.warning(f"[{stem}] → empty or unreadable PDF")
            continue

        metadata = PaperMetadata(
            paper_name=pdf_file.name,
            paper_path=str(pdf_file),
            text_preview=pdf_text[:300],
            num_chars=len(pdf_text),
        )
        save_json(metadata_file, metadata.model_dump())

        truncated = truncate_text(pdf_text, max_chars)
        user_prompt = user_template.replace("{title}", pdf_file.name).replace("{content}", truncated)

        logger.info(f"[{stem}] evaluating...")
        raw = client.generate_structured(system_prompt, user_prompt, temperature=temperature, max_retries=max_retries)
        if raw is None:
            logger.error(f"[{stem}] failed to get valid evaluation")
            continue

        is_relevant = raw.get("is_relevant_domain", False)
        quality_score = raw.get("reward_quality_score")
        if not is_relevant:
            passes = False
            quality_score = None
        else:
            passes = bool(raw.get("passes_quality") and quality_score is not None and quality_score >= threshold)

        result = EvaluationResult(
            paper_name=pdf_file.name,
            paper_path=str(pdf_file),
            is_relevant_domain=is_relevant,
            has_reward_function=raw.get("has_reward_function"),
            reward_is_single_well_defined=raw.get("reward_is_single_well_defined"),
            reward_quality_score=quality_score,
            task_clearly_defined=raw.get("task_clearly_defined"),
            passes_quality=passes,
            reasoning=raw.get("reasoning", ""),
            model_used=model_name,
        )
        save_json(evaluation_file, result.model_dump())
        results.append(result)
        status = "accepted" if result.passes_quality else "rejected"
        logger.info(f"[{stem}] → {status}")

    return results, len(papers), skipped


def print_evaluation_stats(results: list[EvaluationResult], total: int, skipped: int) -> None:
    out_of_scope = sum(1 for r in results if not r.is_relevant_domain)
    in_scope = [r for r in results if r.is_relevant_domain]
    accepted = sum(1 for r in in_scope if r.passes_quality)
    rejected = len(in_scope) - accepted

    print()
    print("📊 Evaluation Results")
    print(f"  Total papers: {total}")
    print(f"  Already cached: {skipped}")
    print(f"  Out of scope: {out_of_scope}")
    print(f"  In scope: {len(in_scope)}")
    if in_scope:
        print(f"    Accepted: {accepted}")
        print(f"    Rejected (quality): {rejected}")
    print()
