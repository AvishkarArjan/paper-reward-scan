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
| **Qwen3-14B** (dense, Apache 2.0) | `-m Qwen/Qwen3-14B` (default) | ~8-10 GB |
| Qwen3-8B (dense, Apache 2.0) | `-m Qwen/Qwen3-8B` | ~5-6 GB |
| Qwen2.5-1.5B-Instruct (small/fast) | `-m Qwen/Qwen2.5-1.5B-Instruct` | ~2 GB |

On a 16 GB card, Qwen3-14B in 4-bit is the best-quality fit and is the default.
(Larger models like Qwen3.8-27B or GLM-4.7-Flash do **not** fit standalone in 4-bit
on 16 GB because all weights must be resident; use them via vLLM/CPU-offload only.)

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
prs stats                      # timing stats per step/paper (pretty rich tables)
```

Every pipeline command logs timestamps + durations to `logs/prs.log` (console +
file) and accumulates machine-readable stats in `logs/stats.json`. `prs stats`
renders those runs as tables; use `prs stats -n 5` to show only the last 5.

Evaluate a single paper / force re-run:

```bash
prs evaluate PAPERS/my-paper.pdf
prs evaluate -f                # ignore cached results, re-run everything
```

`prs evaluate` and `prs extract` read the OCR'd markdown by default. Papers with
no OCR output are skipped; pass `-r/--raw` to force the raw pypdf fallback.

Every command accepts `-m <model>` and most accept `-f`. Skip flags take effect
per stage: `prs evaluate` and `prs extract` cache per-paper results.

## Output

```
output/
├── ocr/{paper}.md                # clean OCR text (text + LaTeX formulas)
├── evaluations/{paper}.json      # pass/reject + quality score per paper
├── extractions/{paper}.json      # extracted reward code + context
├── logs/
│   ├── prs.log                   # human-readable run log (timestamps + durations)
│   └── stats.json                # structured timing stats, shown by `prs stats`
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