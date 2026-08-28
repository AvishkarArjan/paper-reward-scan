# Paper Reward Scan (prs)

Builds an SFT dataset from UAV reinforcement-learning papers. Each PDF is first
OCR'd (text + formulas), an LLM keeps only papers with a high-quality reward
function, then extracts the reward function into standalone Python code.

```
PAPERS/*.pdf  →  ocr  →  evaluate  →  extract  →  compile  →  output/dataset/
```

## Install

```bash
conda create -n prs python=3.12 -y && conda activate prs
pip install -e ".[openai]"
```

Needs an **NVIDIA GPU**. Models run locally in 4-bit.

## 1. Add papers

Drop your PDFs into `PAPERS/`.

## 2. Pick a model

Set the default in `configs/settings.yaml` (`model.default`) or pass `-m` per run.

| Model | Command | VRAM (4-bit) |
|---|---|---|
| GLM-4.7-Flash (30B-A3B MoE, MIT) | `-m zai-org/GLM-4.7-Flash` | ~18 GB |
| Qwen3.8-27B (dense 27B, Apache 2.0) | `-m Qwen/Qwen3.8-27B` | ~15 GB |

On a 16 GB card, use Qwen3.8-27B (or GLM with CPU offload — slower).

> Faster: serve with vLLM and point the pipeline at it — see below.

## 3. Run the pipeline

```bash
# Full pipeline: ocr → evaluate → extract → compile
prs run-all

# Same, but with a specific model (ocr step is unchanged)
prs run-all -m Qwen/Qwen3.8-27B
prs run-all -m zai-org/GLM-4.7-Flash

# Use a vLLM server you already started (keeps the model loaded, faster)
prs run-all -m vllm/Qwen/Qwen3.8-27B
```

Step by step:

```bash
prs ocr                        # OCR papers → clean text + LaTeX formulas
prs evaluate                   # screen papers, keep only good reward functions
prs extract                    # extract reward code from accepted papers
prs compile                    # merge all SFT pairs into one dataset
prs status                     # show how far along the pipeline you are
```

Evaluate a single paper / force re-run:

```bash
prs evaluate PAPERS/my-paper.pdf
prs evaluate -f                # ignore cached results, re-run everything
```

Every command accepts `-m <model>` and most accept `-f`. Skip flags take effect
per stage: `prs evaluate` and `prs extract` cache per-paper results.

## Output

```
output/
├── ocr/{paper}.md                # clean OCR text (text + LaTeX formulas)
├── evaluations/{paper}.json      # pass/reject + quality score per paper
├── extractions/{paper}.json      # extracted reward code + context
└── dataset/
    ├── pairs/{paper}.json        # individual SFT pairs (Alpaca format)
    └── compiled/
        ├── compiled.json         # full dataset (JSON array)
        └── compiled.jsonl        # full dataset (JSONL, Unsloth-ready)
```

The `evaluate`/`extract` stages read `ocr/*.md` when present, so run `prs ocr`
first.

## OCR (Unlimited-OCR)

`prs ocr` runs papers through [Unlimited-OCR](https://github.com/baidu/Unlimited-OCR)
(text, tables, and formulas as LaTeX). Two engines (`ocr.engine` in
`configs/settings.yaml`):

**transformers (default)** — in-process, needs a *separate* env with the pinned
deps (they conflict with this project's newer transformers):

```bash
conda create -n prs-ocr python=3.12 -y && conda activate prs-ocr
pip install -r requirements-ocr.txt
pip install -e .
prs ocr
```

**vllm (recommended if you serve models)** — OCR over an OpenAI-compatible
Unlimited-OCR server (e.g. `docker run --gpus all -p 8001:8001 vllm/vllm-openai:unlimited-ocr`);
set `ocr.engine: vllm` and `ocr.vllm.base_url` accordingly.

## Config

`configs/settings.yaml` — model, quality threshold, temperature, paths, OCR engine.

Prompts are one file per step, edit freely:
- `configs/prompts/ocr.yaml` — what Unlimited-OCR should extract
- `configs/prompts/evaluate.yaml` — screening rules (domain, quality gates)
- `configs/prompts/extract.yaml` — reward function extraction (fix names, schemas, etc.)

## Run with vLLM

Serve the model with vLLM (install it in its own env so it doesn't touch this one):

```bash
conda create -n vllm python=3.12 -y && conda activate vllm
pip install vllm

# Qwen3.8-27B
vllm serve Qwen/Qwen3.8-27B --max-model-len 32768 --gpu-memory-utilization 0.9

# or GLM-4.7-Flash
vllm serve zai-org/GLM-4.7-Flash --max-model-len 32768 --gpu-memory-utilization 0.9
```

Then run the pipeline against it (the model stays loaded between commands):

```bash
conda activate prs
pip install -e ".[openai]"                # vLLM client uses the OpenAI SDK
prs run-all -m vllm/Qwen/Qwen3.8-27B
prs run-all -m vllm/zai-org/GLM-4.7-Flash
```

- If you serve under a custom `--served-model-name foo`, use `-m vllm/foo`.
- To make it the default, set `model.default: "vllm/Qwen/Qwen3.8-27B"` in `configs/settings.yaml`.
- The server address lives in `configs/settings.yaml` (`vllm.base_url`, default `http://localhost:8000/v1`).
- Quantized weights work too: `vllm serve Qwen/Qwen3.8-27B-AWQ --quantization awq`.

## Tips

- First run downloads the model weights into `models/` (takes a while).
- If you run out of VRAM, close other GPU processes; with vLLM use `--gpu-memory-utilization 0.9`.
- Without a GPU or vLLM, any OpenAI/Gemini model works via API prefixes: `-m openai/gpt-4o`, `-m google/gemini-2.5-flash` (set the matching API key).