import re
import io
import base64
import logging
import tempfile
import contextlib
import shutil
from pathlib import Path

from .utils import (
    load_yaml, get_paper_files,
    compute_file_hash, load_content_registry, save_content_registry,
)

logger = logging.getLogger(__name__)

_DET_RE = re.compile(r"<\|det\|>([^<\s]+)(?:\s*\[[^\]]*\])?\s*<\|/det\|>(.*)", re.DOTALL)


def remove_det(raw: str) -> str:
    """Strip <|det|>type [bbox]<|/det|> markers, grouping blocks.

    Lines in the same block are joined with \\n, separate blocks with \\n\\n.
    Image blocks are dropped.
    """
    blocks: list[list[str]] = []
    cur: list[str] | None = None
    for line in raw.splitlines():
        line = line.rstrip()
        if not line:
            continue
        m = _DET_RE.match(line)
        if m:
            category, content = m.group(1).strip(), m.group(2).strip()
            if category == "image":
                continue
            if cur is not None:
                blocks.append(cur)
            cur = [content] if content else []
            continue
        if cur is None:
            cur = []
        cur.append(line)
    if cur is not None:
        blocks.append(cur)
    return "\n\n".join("\n".join(b) for b in blocks).strip()


def pdf_to_images(pdf_path: Path, dpi: int = 300) -> tuple[str, list[str]]:
    """Render PDF pages to PNGs. Returns (temp_dir, [png paths]); caller removes temp_dir."""
    try:
        import pymupdf
    except ImportError:
        import fitz as pymupdf

    doc = pymupdf.open(str(pdf_path))
    tmp_dir = tempfile.mkdtemp(prefix="pdf_ocr_")
    mat = pymupdf.Matrix(dpi / 72, dpi / 72)
    paths = []
    try:
        for i, page in enumerate(doc):
            out = str(Path(tmp_dir) / f"page_{i + 1:04d}.png")
            page.get_pixmap(matrix=mat).save(out)
            paths.append(out)
    finally:
        doc.close()
    return tmp_dir, paths


def read_result_files(output_path: Path) -> str:
    if not output_path.exists():
        return ""
    parts = []
    for f in sorted(output_path.iterdir()):
        if f.is_file() and f.suffix.lower() in (".md", ".txt", ".json", ".jsonl"):
            try:
                parts.append(f.read_text(errors="ignore"))
            except OSError:
                continue
    return "\n\n".join(parts).strip()


def encode_image(image_path: str) -> dict:
    ext = Path(image_path).suffix.lower()
    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else f"image/{ext.lstrip('.')}"
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}


class BaseOCRClient:
    def ocr_pdf(self, pdf_path: Path, prompt: str, output_path: Path, dpi: int = 300,
                image_size: int = 1024, max_length: int = 32768,
                no_repeat_ngram_size: int = 35, ngram_window: int = 1024) -> str:
        """OCR a PDF to clean markdown text (formulas as LaTeX). Return '' on failure."""
        raise NotImplementedError

    def close(self) -> None:
        pass


class TransformersOCRClient(BaseOCRClient):
    """In-process inference via AutoModel (configs/prompts: baidu/Unlimited-OCR)."""

    provider_name = "transformers"

    def __init__(self, model_name: str = "baidu/Unlimited-OCR", cache_dir: str = "models"):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model = None
        self.tokenizer = None

    def _load(self):
        if self.model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        cache_path = Path(self.cache_dir).resolve()
        cache_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading OCR model {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True, cache_dir=str(cache_path)
        )
        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_safetensors=True,
            torch_dtype=torch.bfloat16,
            cache_dir=str(cache_path),
        )
        self.model = self.model.eval().cuda()

    def ocr_pdf(self, pdf_path, prompt, output_path, dpi=300, image_size=1024,
                max_length=32768, no_repeat_ngram_size=35, ngram_window=1024) -> str:
        self._load()
        tmp_dir, image_files = pdf_to_images(pdf_path, dpi)
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.model.infer_multi(
                    self.tokenizer,
                    prompt=prompt,
                    image_files=image_files,
                    output_path=str(output_path),
                    image_size=image_size,
                    max_length=max_length,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    ngram_window=ngram_window,
                    save_results=True,
                )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        return remove_det(read_result_files(output_path))

    def close(self):
        self.model = None
        self.tokenizer = None
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class VLLMOCRClient(BaseOCRClient):
    """Local vLLM server serving Unlimited-OCR (OpenAI-compatible API)."""

    provider_name = "vllm"

    def __init__(self, model_name: str = "Unlimited-OCR",
                 base_url: str = "http://localhost:8001/v1",
                 api_key: str = "EMPTY",
                 images_config: dict | None = None,
                 custom_logit_processor: str | None = None,
                 custom_params: dict | None = None):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.images_config = images_config or {"image_mode": "base"}
        self.custom_logit_processor = custom_logit_processor
        self.custom_params = custom_params

    def ocr_pdf(self, pdf_path, prompt, output_path, dpi=300, image_size=1024,
                max_length=32768, no_repeat_ngram_size=35, ngram_window=1024) -> str:
        tmp_dir, image_files = pdf_to_images(pdf_path, dpi)
        try:
            try:
                from openai import OpenAI
            except ImportError:
                raise RuntimeError("openai SDK not installed. Run: pip install -e '.[openai]'")

            client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=3600)
            content = [{"type": "text", "text": prompt}] + [encode_image(p) for p in image_files]

            extra_body = {"skip_special_tokens": False, "images_config": self.images_config}
            if self.custom_logit_processor:
                extra_body["custom_logit_processor"] = self.custom_logit_processor
            if self.custom_params:
                extra_body["custom_params"] = self.custom_params

            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": content}],
                temperature=0,
                max_tokens=max_length,
                extra_body=extra_body,
            )
            return remove_det(response.choices[0].message.content or "")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def create_ocr_client(cfg: dict) -> BaseOCRClient:
    if cfg.get("engine") == "vllm":
        v = cfg.get("vllm", {})
        return VLLMOCRClient(
            model_name=v.get("model", "Unlimited-OCR"),
            base_url=v.get("base_url", "http://localhost:8001/v1"),
            api_key=v.get("api_key", "EMPTY"),
            images_config=v.get("images_config", {"image_mode": "base"}),
            custom_logit_processor=v.get("custom_logit_processor"),
            custom_params=v.get("custom_params"),
        )
    return TransformersOCRClient(
        model_name=cfg.get("model", "baidu/Unlimited-OCR"),
        cache_dir=cfg.get("cache_dir", "models"),
    )


def ocr_papers(paper_path: Path, settings: dict, force: bool = False) -> tuple[int, int]:
    output_dir = Path(settings["paths"]["output_dir"])
    ocr_dir = output_dir / "ocr"
    raw_dir = ocr_dir / "raw"
    ocr_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    ocr_cfg = settings.get("ocr", {})
    prompt = load_yaml(Path(ocr_cfg.get("prompt_file", "configs/prompts/ocr.yaml")))["prompt"]

    papers = get_paper_files(paper_path)
    if not papers:
        logger.warning(f"No PDF files found at {paper_path}")
        return 0, 0

    client = create_ocr_client(ocr_cfg)
    processed = 0
    skipped = 0
    try:
        for pdf_file in papers:
            stem = pdf_file.stem
            clean_file = ocr_dir / f"{stem}.md"

            if clean_file.exists() and not force:
                skipped += 1
                logger.info(f"[{stem}] → cached OCR")
                continue

            file_hash = compute_file_hash(pdf_file)
            if not force:
                registry = load_content_registry(output_dir)
                if file_hash in registry:
                    prev = ocr_dir / f"{registry[file_hash]}.md"
                    if prev.exists() and prev != clean_file:
                        shutil.copyfile(prev, clean_file)
                        skipped += 1
                        logger.info(f"[{stem}] → OCR identical to {registry[file_hash]}.pdf, copied")
                        continue

            logger.info(f"[{stem}] running Unlimited-OCR...")
            text = client.ocr_pdf(
                pdf_file,
                prompt,
                output_path=raw_dir / stem,
                dpi=ocr_cfg.get("dpi", 300),
                image_size=ocr_cfg.get("image_size", 1024),
                max_length=ocr_cfg.get("max_length", 32768),
                no_repeat_ngram_size=ocr_cfg.get("no_repeat_ngram_size", 35),
                ngram_window=ocr_cfg.get("ngram_window", 1024),
            )
            if not text:
                logger.warning(f"[{stem}] → empty OCR output, skipping")
                continue

            clean_file.write_text(text)
            registry = load_content_registry(output_dir)
            registry[file_hash] = stem
            save_content_registry(output_dir, registry)
            processed += 1
            logger.info(f"[{stem}] → OCR done ({len(text) / 1024:.1f} KB)")
    finally:
        client.close()

    print()
    print("📄 OCR Results")
    print(f"  Papers: {len(papers)}")
    print(f"  OCR'd: {processed}")
    print(f"  Cached: {skipped}")
    print()

    return processed, skipped