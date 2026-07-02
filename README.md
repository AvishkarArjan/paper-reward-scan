# Paper Reward Scan (prs)

Build SFT datasets from UAV RL papers using an LLM extractor-critic pipeline.

```
papers/*.pdf  →  evaluate (critic)  →  extract  →  compile  →  dataset/
```

## Quickstart

```bash
pip install -e ".[openai]" && huggingface-cli login
# Drop PDFs into papers/, then:
prs evaluate && prs extract && prs compile
```

## Setup

### Requirements

- Python 3.10+
- **GPU**: NVIDIA with 8GB+ VRAM + CUDA (for local models)
- **No GPU**: Just a HuggingFace token (free Inference API)

### Install

```bash
git clone <repo-url> && cd paper-reward-scan
conda create -n prs python=3.12 -y && conda activate prs
pip install -e .
huggingface-cli login
```

### Optional: Commercial API support

```bash
pip install -e ".[openai]"        # OpenAI / xAI (Grok)
pip install -e ".[google]"        # Google Gemini
pip install -e ".[openai,google]" # All
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="..."
export XAI_API_KEY="..."
```

## Commands

### `prs evaluate` — Shortlist papers with quality reward functions

Reads each PDF, runs an LLM critic to check domain relevance and reward quality.

```bash
# Evaluate all PDFs in papers/
prs evaluate

# Evaluate a single paper
prs evaluate papers/my-paper.pdf

# Use a specific model (default: google/gemini-2.5-flash)
prs evaluate -m openai/gpt-4o
prs evaluate -m hf-api/Qwen/Qwen2.5-7B-Instruct   # no GPU needed
prs evaluate -m google/gemini-2.5-flash

# Force re-evaluate (ignore cache)
prs evaluate -f

# Combined
prs evaluate papers/my-paper.pdf -m openai/gpt-4o -f
```

**Per-paper output** (`output/evaluations/{stem}.json`):
```json
{
  "paper_name": "my-paper.pdf",
  "is_relevant_domain": true,
  "has_reward_function": true,
  "reward_is_single_well_defined": true,
  "reward_quality_score": 8,
  "task_clearly_defined": true,
  "passes_quality": true,
  "reasoning": "The paper proposes a DDPG-based approach for autonomous UAV navigation...",
  "model_used": "mistralai/Mistral-7B-Instruct-v0.3"
}
```

Papers must pass **all** gates to be `passes_quality: true`:
1. **Domain relevance** — Platform must be a **flying vehicle** (UAV/drone/quadcopter/VTOL). Ground robots, cars, underwater vehicles are rejected.
2. **Has explicit reward function** — Mathematical expression, pseudocode, or algorithmic description.
3. **Single scalar reward** — Multiple terms OK if combined into one scalar (not vague multi-objective).
4. **Quality score ≥ threshold** — The critic rates clarity, sensibility, completeness, and implementability 1-10. Default threshold: 7 (set in `configs/settings.yaml`).
5. **Task clearly defined** — Goals, state/action space, and success criteria described.

**Rejected papers** (`is_relevant_domain: false` or `passes_quality: false`) are skipped by subsequent steps.

### `prs extract` — Extract reward functions from accepted papers

For each `passes_quality` paper, the extractor reads the full text and produces a standalone Python reward function plus an SFT pair.

```bash
prs extract                              # extract from all accepted papers
prs extract papers/specific-paper.pdf    # extract from a single paper
prs extract -m openai/gpt-4o             # use a specific model
prs extract -f                           # force re-extract
```

**Per-paper output** (`output/extractions/{stem}.json`):
```json
{
  "paper_name": "my-paper.pdf",
  "task_description": "Autonomous UAV navigation in urban canyon...",
  "environment_context": "State: position, velocity, lidar scans. Action: 3D velocity commands...",
  "components": ["Collision penalty", "Goal proximity reward", "Energy regularization"],
  "reward_function_code": "def compute_reward(state, action):\n    ...",
  "model_used": "mistralai/Mistral-7B-Instruct-v0.3"
}
```

**SFT pair output** (`output/dataset/pairs/{stem}.json`):
```json
{
  "instruction": "Given the UAV task description and environment context below, generate the reward function as a standalone Python function.",
  "input": "Task: autonomous quadrotor landing...\nEnvironment: Gymnasium...",
  "output": "```python\ndef compute_reward(state, action):\n    ...\n```"
}
```

### `prs compile` — Merge SFT pairs into a single dataset

Aggregates all SFT pairs from `output/dataset/pairs/` into two formats:

```bash
prs compile
```

- `output/dataset/compiled/compiled.json` — JSON array, entry per paper
- `output/dataset/compiled/compiled.jsonl` — JSON Lines, one record per line (ready for Unsloth)

### `prs run-all` — Full pipeline (evaluate → extract → compile)

```bash
prs run-all
prs run-all -m openai/gpt-4o
prs run-all -f
```

Effectively runs `prs evaluate`, `prs extract`, `prs compile` in sequence with shared flags.

### `prs status` — Pipeline progress overview

```bash
prs status
```

Example output:
```
📊 Pipeline Status
  Directory: papers/ (7 PDFs)
  Evaluated: 7/7
    Accepted: 5  |  Rejected (quality): 2
  Extracted: 5/5
  Compiled: 5 samples (output/dataset/compiled/)
```

## Models

| Model | CLI prefix | GPU? | Key needed? |
|---|---|---|---|
| HuggingFace local (4-bit) | `-m hf-org/model-name` | ✅ Yes | — |
| HF Inference API (free) | `-m hf-api/org/model-name` | ❌ No | HF token |
| OpenAI | `-m openai/gpt-4o` | ❌ No | `OPENAI_API_KEY` |
| Google Gemini | `-m google/gemini-2.5-flash` | ❌ No | `GOOGLE_API_KEY` |
| xAI Grok | `-m xai/grok-3` | ❌ No | `XAI_API_KEY` |

**Default** (`configs/settings.yaml`): `google/gemini-2.5-flash` (API key required).

### Google model rate limits (free tier)

| CLI model ID | Model | Requests/min | TPM | Requests/day |
|---|---|---|---|---|
| `google/gemini-2.5-flash-lite` | Gemini 2.5 Flash Lite | 15 | 250K | 500 |
| `google/gemini-3.1-flash-lite-preview` | Gemini 3.1 Flash Lite (preview) | 15 | 250K | 500 |
| `google/gemma-4-26b-a4b-it` | Gemma 4 26B | 15 | Unlimited | 1.5K |
| `google/gemma-4-31b-it` | Gemma 4 31B | 15 | Unlimited | 1.5K |

Set the matching `rate_limits.google` in `configs/settings.yaml` if you switch models.

**No-GPU example**:
```bash
prs evaluate -m hf-api/Qwen/Qwen2.5-7B-Instruct
```
Uses HuggingFace's free Inference API — just need `huggingface-cli login`.

## Configuration

### `configs/settings.yaml`

| Key | Default | Description |
|---|---|---|
| `model.default` | `google/gemini-2.5-flash` | Model used when no `-m` flag |
| `model.hf_cache_dir` | `models/` | HuggingFace cache directory |
| `evaluation.quality_threshold` | `7` | Minimum reward quality score (1-10) |
| `evaluation.max_paper_chars` | `20000` | Characters fed to LLM per paper |
| `evaluation.temperature` | `0.01` | LLM sampling temperature |
| `evaluation.max_retries` | `3` | Retries on malformed LLM output |
| `paths.papers_dir` | `papers/` | PDF input directory |
| `paths.output_dir` | `output/` | Output root |
| `rate_limits.google` | `5` | Gemini API requests per minute |
| `rate_limits.openai` | `60` | OpenAI API requests per minute |
| `rate_limits.xai` | `60` | xAI API requests per minute |
| `rate_limits.hf-api` | `30` | HF Inference API requests per minute |

### Prompt customization (`configs/prompts/`)

- **`evaluate.yaml`** — The critic system prompt. Defines domain relevance rules, quality criteria, and JSON schema. Edit to tighten/loosen acceptance gates.
- **`extract.yaml`** — The extractor system prompt. Instructs the model how to read the paper and produce a standalone reward function.

## Output structure

```
output/
├── metadata/{stem}.json            # PDF text cache (avoids re-reading)
├── evaluations/{stem}.json         # Critic verdict per paper
├── extractions/{stem}.json         # Extracted reward + context
└── dataset/
    ├── pairs/{stem}.json            # Individual SFT pairs (Alpaca format)
    └── compiled/
        ├── compiled.json            # Full dataset (JSON array)
        └── compiled.jsonl           # Full dataset (JSONL — Unsloth-ready)
```

## How it works

1. **Evaluate**: LLM critic reads each paper, checks domain (flying vehicle only), identifies reward functions, scores quality 1-10, and decides pass/reject.
2. **Extract**: For accepted papers, an LLM extracts the reward function, task description, environment context, and components into a standalone Python function.
3. **Compile**: All SFT pairs are merged into a single dataset in Unsloth Alpaca format (instruction/input/output).
