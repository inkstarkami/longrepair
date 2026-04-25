# LongRepair: Enhancing Multi-Hunk Automated Program Repair via Reasoning-Oriented Imitation and Reinforcement Learning

This repository provides scripts for model training and patch generation, used in paper **LongRepair: Enhancing Multi-Hunk Automated Program Repair via Reasoning-Oriented Imitation and Reinforcement Learning**.

---

## Repository Structure

```
bug_repair_paper_clean/
├── README.md
├── requirements.txt
├── dataset/
│   ├── defects4j_benchmark.jsonl       # 835-bug Defects4J evaluation set
│   └── defects4j_filename_map.json     # Per-bug source path / bug-line map
├── training/                           # verl recipe (drop into recipe/<name>/)
│   ├── data/
│   │   ├── sft_train.parquet           # SFT training set
│   │   ├── sft_val.parquet             # SFT validation set
│   │   ├── grpo_train.parquet          # GRPO training set
│   │   └── grpo_val.parquet            # GRPO validation set
│   ├── reward_function.py              # Format + diff + coverage reward
│   ├── train_sft.sh                    # SFT entry point (FSDP)
│   └── train_grpo.sh                   # GRPO entry point (vLLM rollout)
└── inference/
    ├── scripts/
    │   ├── run_generation.sh           # Sharded patch generation
    │   └── run_validation.sh           # Auto-discovers and validates outputs
    └── src/
        ├── generate_patches.py         # HF / PEFT inference loop
        ├── prompter.py                 # System / user prompt templates
        └── validate_patch.py           # Defects4J test runner + parser
```

---

## 1. Prerequisites

### 1.1 Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 1.2 Install Defects4J

Follow the official guide: https://github.com/rjust/defects4j

### 1.3 Install verl

Needed for the training pipeline. Install from source so that the
`vllm` / `ray` / CUDA versions match your host:
https://github.com/volcengine/verl

Then drop the `training/` folder into your verl checkout:

```bash
cp -r training /path/to/verl/recipe/bug_repair
```

---

## 2. Training

The training scripts are written as a verl recipe. After Section 1.3 the
recipe (including the four parquets under `training/data/`) lives at
`verl/recipe/bug_repair/`. Run all commands below from the verl
repository root.

### 2.1 SFT

```bash
BASE_MODEL=/path/to/base_model \
SFT_CHECKPOINT_DIR=$PWD/checkpoints/sft \
bash recipe/bug_repair/train_sft.sh
```

After completion, `${SFT_CHECKPOINT_DIR}/merged_model` is the artifact
consumed by GRPO.

### 2.2 GRPO

```bash
SFT_MERGED_MODEL=$PWD/checkpoints/sft/merged_model \
GRPO_CHECKPOINT_DIR=$PWD/checkpoints/grpo \
bash recipe/bug_repair/train_grpo.sh
```

After completion, `${GRPO_CHECKPOINT_DIR}/merged_model` can be plugged
straight into Section 3 as `MODEL_PATH`.

---

## 3. Inference + Validation

### 3.1 Generate Patches

One process runs through the entire benchmark with the given model.

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL_PATH=/path/to/merged_model \
OUTPUT_PATH=$PWD/results/run.jsonl \
bash inference/scripts/run_generation.sh
```

`MODE` (default `with_context`) selects the prompt variant:
`with_context` includes the file skeleton, `without_context` only includes
the buggy functions.

### 3.2 Validate Patches

```bash
bash inference/scripts/run_validation.sh $PWD/results/run.jsonl
```

Produces `$PWD/results/run_result.jsonl` with one line per bug
(`{bug_id, passed, results}`). Generations whose decoded length is at or
above `MAX_PATCH_TOKENS` are dropped before validation.

To use a specific conda environment, pass `PYTHON_BIN`:

```bash
PYTHON_BIN=$(conda run -n myenv which python) \
bash inference/scripts/run_validation.sh $PWD/results/run.jsonl
```

---
