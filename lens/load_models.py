"""Load Pythia checkpoints with their true per-step weights, in float32.

Two traps make a naive ``from_pretrained(..., revision="stepN")`` wrong here:

* ``EleutherAI/pythia-2.8b`` ships a stray consolidated ``model.safetensors``
  holding main's fully-trained weights on every step branch except step143000,
  and ``transformers`` prefers it over the branch's correct per-step shards
  (HF discussion EleutherAI/pythia-2.8b#5) -- so every "checkpoint" silently
  loads the final model. Steps 40000-130000 additionally ship no per-step
  shards at all; their true weights exist only in ``pytorch_model.bin``.
  ``resolve_model_source`` routes around both and refuses the consolidated
  fallback.
* Loading must be float32. That is exact at every size: 70m/160m/410m/1.4b
  ship fp32, and 2.8b/6.9b ship fp16, whose upcast is lossless. float16
  loading is unsafe: the transported residual ``J_l @ h`` can exceed fp16's
  65504 ceiling, and inf -> LayerNorm -> NaN logits makes a rank readout
  return 1 for every token -- a silent perfect score instead of a crash.

Run as a script to prefetch checkpoints into the Hugging Face cache:

    python -m lens.load_models --models 2.8b,6.9b --steps 0,143000
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import torch
from huggingface_hub import list_repo_files, snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
SAFE_DIR_ROOT = REPO_ROOT / ".cache" / "pythia_safe"

MODELS = ["70m", "160m", "410m", "1.4b", "2.8b", "6.9b"]
CHECKPOINT_STEPS = [0, 16, 256, 512, 1000, 2000, 3000, 4000,
                    16000, 40000, 70000, 100000, 130000, 143000]

BROKEN_REPOS = {"EleutherAI/pythia-2.8b"}
CLEAN_REVISIONS = {None, "main", "step143000"}
STRAY_FILE = "model.safetensors"
SAFE_PATTERNS = [
    "config.json", "generation_config.json", "tokenizer*", "special_tokens*",
    "model-0000*", "model.safetensors.index.json",
]
BIN_PATTERNS = [
    "config.json", "generation_config.json", "tokenizer*", "special_tokens*",
    "pytorch_model*",
]
_probe_cache: dict[tuple[str, str], bool] = {}


def model_id_for(size: str) -> str:
    return f"EleutherAI/pythia-{size}"


def resolve_model_source(model_id: str, revision: str | None) -> tuple[str, str | None]:
    """Return ``(path_or_id, revision)`` that loads the revision's true weights."""
    if revision in CLEAN_REVISIONS:
        return model_id, revision
    if model_id not in BROKEN_REPOS and not _serves_stray_monolith(model_id, revision):
        return model_id, revision
    snapshot = Path(snapshot_download(model_id, revision=revision, allow_patterns=SAFE_PATTERNS))
    if not (snapshot / "model.safetensors.index.json").exists():
        # Some branches (2.8b steps 40000-130000) have no shards at all; the
        # true per-step weights live only in pytorch_model.bin there.
        snapshot = Path(snapshot_download(model_id, revision=revision, allow_patterns=BIN_PATTERNS))
    safe = SAFE_DIR_ROOT / f"{model_id.replace('/', '_')}@{revision}"
    safe.mkdir(parents=True, exist_ok=True)
    for entry in snapshot.iterdir():
        if entry.name == STRAY_FILE:
            continue
        link = safe / entry.name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(entry.resolve())
    has_weights = (safe / "model.safetensors.index.json").exists() or any(
        safe.glob("pytorch_model*.bin")
    )
    if not has_weights:
        raise RuntimeError(
            f"{model_id}@{revision}: neither sharded safetensors nor a "
            f"pytorch_model.bin in the snapshot; refusing to fall back to "
            f"the stray consolidated {STRAY_FILE}"
        )
    return str(safe), None


def _serves_stray_monolith(model_id: str, revision: str) -> bool:
    """Probe whether the revision lists a consolidated file next to shards.

    That combination is the signature of the bug: transformers will prefer
    the consolidated file. A probe that cannot reach the Hub after retries
    warns and reports False -- add the repo to ``BROKEN_REPOS`` to make the
    safe path unconditional.
    """
    key = (model_id, revision)
    if key in _probe_cache:
        return _probe_cache[key]
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            files = set(list_repo_files(model_id, revision=revision))
            break
        except Exception as error:  # the hub raises many types
            last_error = error
            time.sleep(2**attempt)
    else:
        warnings.warn(
            f"could not inspect {model_id}@{revision} for the stray-weights "
            f"pattern ({last_error!r}); loading through the default path. "
            f"If this repo is affected, add it to BROKEN_REPOS in "
            f"lens/load_models.py.",
            stacklevel=3,
        )
        _probe_cache[key] = False
        return False
    stray = (
        STRAY_FILE in files
        and "model.safetensors.index.json" in files
        and any(n.startswith("model-") and n.endswith(".safetensors") for n in files)
    )
    _probe_cache[key] = stray
    return stray
