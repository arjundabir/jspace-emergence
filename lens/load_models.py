from __future__ import annotations

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


def resolve_model_source(model_id: str, revision: str | None) -> tuple[str, str | None]:
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
        link.unlink(missing_ok=True)
        link.symlink_to(entry.resolve())
    return str(safe), None


def _serves_stray_monolith(model_id: str, revision: str) -> bool:
    key = (model_id, revision)
    if key in _probe_cache:
        return _probe_cache[key]
    files = set(list_repo_files(model_id, revision=revision))
    stray = (
        STRAY_FILE in files
        and "model.safetensors.index.json" in files
        and any(n.startswith("model-") and n.endswith(".safetensors") for n in files)
    )
    _probe_cache[key] = stray
    return stray


def load_model(model_id: str, revision: str | None, device: str | torch.device | None = None):
    resolved = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True)
    source, source_revision = resolve_model_source(model_id, revision)
    hf_model = AutoModelForCausalLM.from_pretrained(
        source, revision=source_revision, dtype=torch.float32
    )
    return hf_model.to(resolved).eval(), tokenizer
