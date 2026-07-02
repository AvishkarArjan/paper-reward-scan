import logging
from pathlib import Path

from .schemas import EvaluationResult, ExtractionResult, SFTPair
from .utils import (
    load_yaml, save_json, load_json, read_pdf,
    get_paper_files, truncate_text,
    compute_file_hash, load_content_registry, save_content_registry,
)
from .models import create_client

logger = logging.getLogger(__name__)


def extract_rewards(
    paper_path: Path,
    settings: dict,
    model_name: str,
    force: bool = False,
) -> list[ExtractionResult]:
    papers_dir = Path(settings["paths"]["papers_dir"])
    output_dir = Path(settings["paths"]["output_dir"])
    evaluations_dir = output_dir / "evaluations"
    extractions_dir = output_dir / "extractions"
    pairs_dir = output_dir / "dataset" / "pairs"

    max_chars = settings["evaluation"]["max_paper_chars"]
    temperature = settings["extraction"]["temperature"]
    max_retries = settings["extraction"]["max_retries"]
    instruction_text = settings["dataset"]["instruction"]

    papers = get_paper_files(paper_path)
    if not papers:
        logger.warning(f"No PDF files found at {paper_path}")
        return []

    prompt_data = load_yaml(Path("configs/prompts/extract.yaml"))
    system_prompt = prompt_data["system"]
    user_template = prompt_data["user"]

    accepted = []
    for pdf_file in papers:
        stem = pdf_file.stem
        evaluation_file = evaluations_dir / f"{stem}.json"
        evaluation = load_json(evaluation_file)
        if evaluation is None:
            logger.info(f"[{stem}] not evaluateed yet, skipping")
            continue
        if not evaluation.get("passes_quality"):
            logger.info(f"[{stem}] did not pass quality check, skipping")
            continue
        accepted.append(pdf_file)

    if not accepted:
        logger.info("No accepted papers to extract from")
        return []

    client = create_client(model_name, settings["model"]["hf_cache_dir"], settings.get("rate_limits"))
    results = []

    for pdf_file in accepted:
        stem = pdf_file.stem
        extraction_file = extractions_dir / f"{stem}.json"
        pair_file = pairs_dir / f"{stem}.json"

        if extraction_file.exists() and not force:
            existing = load_json(extraction_file)
            if existing:
                results.append(ExtractionResult(**existing))
                logger.info(f"[{stem}] → cached")
                continue

        file_hash = compute_file_hash(pdf_file)

        if not force:
            registry = load_content_registry(output_dir)
            if file_hash in registry:
                existing_stem = registry[file_hash]
                existing_extraction = extractions_dir / f"{existing_stem}.json"
                existing_pair = pairs_dir / f"{existing_stem}.json"
                if existing_extraction.exists():
                    existing = load_json(existing_extraction)
                    if existing:
                        results.append(ExtractionResult(**existing))
                        logger.info(f"[{stem}] → content identical to {existing_stem}.pdf, skipped")
                        continue

        pdf_text = read_pdf(pdf_file)
        if not pdf_text:
            logger.warning(f"[{stem}] → empty or unreadable PDF, skipping")
            continue

        truncated = truncate_text(pdf_text, max_chars)
        user_prompt = user_template.replace("{title}", pdf_file.name).replace("{content}", truncated)

        logger.info(f"[{stem}] extracting reward...")
        raw = client.generate_structured(system_prompt, user_prompt, temperature=temperature, max_retries=max_retries)
        if raw is None:
            logger.error(f"[{stem}] failed to extract")
            continue

        import re
        code = raw.get("reward_function_code", "")
        code_match = re.search(r"```python\s*\n(.*?)```", code, re.DOTALL)
        if not code_match:
            code_match = re.search(r"```\s*\n(.*?)```", code, re.DOTALL)
        if code_match:
            clean_code = code_match.group(1).strip()
        else:
            clean_code = code.strip()

        result = ExtractionResult(
            paper_name=pdf_file.name,
            paper_path=str(pdf_file),
            task_description=raw.get("task_description", ""),
            environment_context=raw.get("environment_context", ""),
            components=raw.get("components", []),
            reward_function_code=clean_code,
            model_used=model_name,
        )
        save_json(extraction_file, result.model_dump())
        registry = load_content_registry(output_dir)
        registry[file_hash] = stem
        save_content_registry(output_dir, registry)

        sft_pair = SFTPair(
            instruction=instruction_text,
            input=f"Task: {result.task_description}\n\nEnvironment: {result.environment_context}\n\nComponents: {', '.join(result.components)}",
            output=f"```python\n{result.reward_function_code}\n```",
        )
        save_json(pair_file, sft_pair.model_dump())

        results.append(result)
        logger.info(f"[{stem}] → extracted")

    print()
    print(f"📝 Extraction Results")
    print(f"  Accepted papers: {len(accepted)}")
    print(f"  Extracted: {len(results)}")
    if len(accepted) - len(results) > 0:
        print(f"  Failed: {len(accepted) - len(results)}")
    print()

    return results
