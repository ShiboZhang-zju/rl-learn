#!/usr/bin/env bash
# Minimal GPU restore check for rl-learn.
#
# Goal: after `git clone` / `git pull` on a new GPU machine, this script verifies
# that the experiment can be resumed. It does NOT train, evaluate, rerun any
# completed experiment, or modify model/experiment outputs.
#
# Steps:
#   A. Environment check (Python / Torch / CUDA / GPU / TRL / PEFT / Transformers / Datasets)
#   B. Install Python dependencies (never touches torch)
#   C. Base model availability (HF cache, download only if missing)
#   D. Key checkpoints
#   E. Key data files
#   F. pytest
#   G. PEFT adapter load smoke test

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

BASE_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
INIT_CKPT="outputs/sft_v2_5k_p800/checkpoint-1252"

CHECKPOINTS=(
  "outputs/sft_v2_5k_p800/checkpoint-1252"
  "outputs/sft_v2_5k_p800/checkpoint-1565"
  "outputs/grpo_v1/checkpoint-200"
)

DATA_FILES=(
  "data/processed/grpo_v1_train.jsonl"
  "data/processed/grpo_v1_final_holdout.jsonl"
  "data/processed/v2_answer_only_val.jsonl"
)

GIT_OK="FAIL"
CUDA_OK="FAIL"
ENV_OK="FAIL"
MODEL_OK="FAIL"
CKPT_OK="FAIL"
DATA_OK="FAIL"
TESTS_OK="FAIL"
ADAPTER_OK="FAIL"

section() {
  echo
  echo "==== $* ===="
}

echo "repo: $REPO_ROOT"

# ---------------------------------------------------------------- A. Environment
section "A. Environment check"

python - <<'PY'
import sys

import torch
import transformers
import datasets
import peft
import trl

print("Python:      ", sys.version.split()[0])
print("Torch:       ", torch.__version__)
print("CUDA:        ", torch.version.cuda)
print("CUDA avail:  ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:         ", torch.cuda.get_device_name(0))
print("TRL:         ", trl.__version__)
print("PEFT:        ", peft.__version__)
print("Transformers:", transformers.__version__)
print("Datasets:    ", datasets.__version__)
PY
ENV_PRINT_RC=$?

if [ "$ENV_PRINT_RC" -ne 0 ]; then
  echo "ENV_FAIL: cannot import torch/transformers/datasets/peft/trl"
else
  ENV_OK="OK"
fi

if [ "$ENV_OK" = "OK" ]; then
  if python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    CUDA_OK="OK"
  else
    echo
    echo "CUDA_FAIL: torch.cuda.is_available() == False"
    echo "This project requires a working CUDA GPU. Stopping."
  fi
fi

if [ "$CUDA_OK" != "OK" ]; then
  section "RESULT"
  echo "CUDA unavailable -> abort before touching any dependency."
  exit 1
fi

# ------------------------------------------------------------------- A2. Git
section "A2. Git state"

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "commit: $(git log -1 --oneline)"
  if [ -n "$(git status --porcelain)" ]; then
    echo "note: working tree has local modifications"
  else
    echo "working tree: clean"
  fi
  GIT_OK="OK"

  # Guard against "doc describes state A, but the checkout is state B".
  ANCHOR="$(grep -oE '^[0-9a-f]{40}' EXPERIMENT_STATE.md 2>/dev/null | head -n 1)"
  if [ -n "$ANCHOR" ]; then
    if git cat-file -e "${ANCHOR}^{commit}" 2>/dev/null; then
      if git merge-base --is-ancestor "$ANCHOR" HEAD 2>/dev/null; then
        echo "state doc anchor: IN_SYNC ($ANCHOR)"
      else
        echo "state doc anchor: OUT_OF_SYNC"
        echo "  EXPERIMENT_STATE.md anchor $ANCHOR is not an ancestor of HEAD."
        echo "  The document and the checked-out code describe different states."
        GIT_OK="FAIL"
      fi
    else
      echo "state doc anchor: $ANCHOR (shallow clone, history unavailable -> skipped)"
    fi
  fi
else
  echo "GIT_FAIL: not a git repository"
fi

# ------------------------------------------------------- B. Python dependencies
section "B. Install Python dependencies"

if [ -f requirements-gpu.txt ]; then
  pip install -r requirements-gpu.txt
  PIP_RC=$?
else
  echo "requirements-gpu.txt not found, skipping."
  PIP_RC=0
fi

pip install -e .
EDITABLE_RC=$?

# torch is deliberately not touched by requirements-gpu.txt; re-assert it here.
python -c "import torch; print('torch after install:', torch.__version__, '| cuda:', torch.version.cuda)"

if [ "$PIP_RC" -eq 0 ] && [ "$EDITABLE_RC" -eq 0 ]; then
  ENV_OK="OK"
else
  ENV_OK="FAIL"
  echo "ENV_FAIL: dependency installation failed"
fi

# ---------------------------------------------------------------- C. Base model
section "C. Base model check: $BASE_MODEL"

python - "$BASE_MODEL" <<'PY'
import sys

from transformers import AutoTokenizer

model = sys.argv[1]

try:
    AutoTokenizer.from_pretrained(model, local_files_only=True)
    print("BASE_MODEL_OK (local cache)")
    sys.exit(0)
except Exception as exc:  # noqa: BLE001
    print("local cache miss:", type(exc).__name__)

try:
    AutoTokenizer.from_pretrained(model)
    print("BASE_MODEL_OK (downloaded)")
    sys.exit(0)
except Exception as exc:  # noqa: BLE001
    print("download failed:", type(exc).__name__, exc)
    print("BASE_MODEL_MISSING")
    sys.exit(1)
PY
MODEL_RC=$?

if [ "$MODEL_RC" -eq 0 ]; then
  MODEL_OK="OK"
else
  MODEL_OK="FAIL"
  echo
  echo "Attempting: huggingface-cli download $BASE_MODEL"
  huggingface-cli download "$BASE_MODEL"
  if [ $? -eq 0 ]; then
    MODEL_OK="OK"
  else
    MODEL_OK="FAIL"
    echo
    echo "BASE_MODEL_MISSING"
    echo "Action required: restore the Hugging Face cache for $BASE_MODEL"
    echo "  - copy the HF cache dir to \$HF_HOME/hub (default ~/.cache/huggingface/hub), or"
    echo "  - point the config model_name_or_path at a local model directory."
  fi
fi

# ---------------------------------------------------------------- D. Checkpoints
section "D. Checkpoint check"

CKPT_OK="OK"
for d in "${CHECKPOINTS[@]}"; do
  if [ -d "$d" ]; then
    echo "OK   $d"
  else
    echo "MISS $d"
    CKPT_OK="FAIL"
  fi
done

if [ "$CKPT_OK" != "OK" ]; then
  echo
  echo "CHECKPOINT_NOT_PORTABLE"
  echo "These adapters are expected to be restored by plain git clone/pull (no Git LFS)."
  echo "Do NOT retrain to recreate them. Restore them from a backup/artifact of this repo."
fi

# ---------------------------------------------------------------------- E. Data
section "E. Data check"

DATA_OK="OK"
for f in "${DATA_FILES[@]}"; do
  if [ -f "$f" ]; then
    echo "OK   $f"
  else
    echo "MISS $f"
    DATA_OK="FAIL"
  fi
done

# --------------------------------------------------------------------- F. Tests
section "F. pytest"

pytest -q
TESTS_RC=$?

if [ "$TESTS_RC" -eq 0 ]; then
  TESTS_OK="OK"
  echo "pytest exit code 0"
else
  TESTS_OK="FAIL"
  echo "TESTS_FAIL: pytest exit code $TESTS_RC"
fi

# ------------------------------------------------------------ G. Adapter smoke
section "G. Adapter load smoke test"

if [ "$MODEL_OK" = "OK" ] && [ -d "$INIT_CKPT" ]; then
  python - "$BASE_MODEL" "$INIT_CKPT" <<'PY'
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model, adapter = sys.argv[1], sys.argv[2]

tok = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.float32)
model = PeftModel.from_pretrained(model, adapter, is_trainable=True)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("base model:  ", base_model)
print("adapter:     ", adapter)
print("trainable params:", trainable)
print("ADAPTER_LOAD_OK")
PY
  if [ $? -eq 0 ]; then
    ADAPTER_OK="OK"
  else
    ADAPTER_OK="FAIL"
  fi
else
  echo "skip: base model or init checkpoint unavailable"
  ADAPTER_OK="FAIL"
fi

# ------------------------------------------------------------------- Summary
echo
echo "================================"
echo "RL-LEARN RESTORE CHECK"
printf 'Git:           %s\n' "$GIT_OK"
printf 'CUDA:          %s\n' "$CUDA_OK"
printf 'Environment:   %s\n' "$ENV_OK"
printf 'Base Model:    %s\n' "$MODEL_OK"
printf 'Checkpoint:    %s\n' "$CKPT_OK"
printf 'Data:          %s\n' "$DATA_OK"
printf 'Tests:         %s\n' "$TESTS_OK"
printf 'Adapter Load:  %s\n' "$ADAPTER_OK"
echo

ALL_OK="yes"
for s in "$GIT_OK" "$CUDA_OK" "$ENV_OK" "$MODEL_OK" "$CKPT_OK" "$DATA_OK" "$TESTS_OK" "$ADAPTER_OK"; do
  [ "$s" = "OK" ] || ALL_OK="no"
done

if [ "$ALL_OK" = "yes" ]; then
  echo "READY_TO_RESUME"
  echo "Next: GRPO-V2"
  echo "================================"
  echo
  echo "Do not rerun completed experiments. Read EXPERIMENT_STATE.md before continuing."
  exit 0
fi

echo "RESTORE_CHECK_FAILED"
echo "Do NOT retrain any completed experiment to 'fix' a failed check."
echo "Read EXPERIMENT_STATE.md and report the failing item(s) first."
echo "================================"
exit 1
