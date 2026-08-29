"""Shared machinery for the six J-lens evaluations.

Every evaluation follows the same protocol: for each fitted lens, read the
residual out at a task-defined prompt position, decode it through the J-lens
and through the logit-lens (J = I) baseline, and score the task's intermediate
concepts by one-based rank over the vocabulary. Metrics are pass@k over
intermediates with min rank across fitted layers, and normalized log-AUC on
k in {1, 2, 5, 10, 20, 50, 100}. A capability control records whether the
checkpoint can produce the task's answer at all.

A task supplies its dataset slug, its readout-position logic, and (for
order-of-operations) a synonym expander; everything else lives here. One JSON
artifact per lens is written under ``results/evals/<model>/<step>/``, plus a
gzipped CSV of every (item, intermediate, surface, layer, method) score, and
``emergence_summary_<eval>.csv`` under ``results/evals/``.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch

import jlens
from jlens.hooks import ActivationRecorder

from lens.load_lens import FITS_ROOT, load_lens, pythia_layout
from lens.load_models import REPO_ROOT, load_model

DATA_ROOT = REPO_ROOT / "jacobian-lens" / "data" / "evaluations"
RESULTS_ROOT = REPO_ROOT / "results" / "evals"

PAPER_K_VALUES = [1, 2, 5, 10, 20, 50, 100]
PAPER_LAYER_COUNT = 25
SCORE_BATCH_SIZE = 8
RANDOM_SEED = 0
METHODS = ["jacobian", "logit"]
TOKENS_PER_STEP = 1024 * 2048

PARAM_COUNTS = {
    "EleutherAI/pythia-70m": 70_426_624,
    "EleutherAI/pythia-160m": 162_322_944,
    "EleutherAI/pythia-410m": 405_334_016,
    "EleutherAI/pythia-1.4b": 1_414_647_808,
    "EleutherAI/pythia-2.8b": 2_775_208_960,
    "EleutherAI/pythia-6.9b": 6_857_302_016,
}

K_VALUES = np.asarray(PAPER_K_VALUES, dtype=np.int64)
LOG_K_VALUES = np.log(K_VALUES.astype(np.float64))


@dataclass(frozen=True)
class Task:
    """What distinguishes one evaluation from the other five."""

    name: str
    slug: str
    readout_mode: str
    prepare_item: Callable[[int, dict, object], dict]
    expand_synonyms: Callable[[str], list[str]] = lambda intermediate: [intermediate]


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def discover_lenses(lens_root: Path) -> list[Path]:
    """Every fitted lens under ``lens_root``, sorted by path."""
    if not lens_root.is_dir():
        raise FileNotFoundError(f"Lens root does not exist: {lens_root}")
    return sorted(path.resolve() for path in lens_root.rglob("*_jlens.pt") if path.is_file())


def identity_from_lens_filename(path: Path) -> tuple[str, str, int]:
    match = re.fullmatch(r"(?P<model_part>.+)_(?P<revision>step(?P<step>\d+))_jlens\.pt", path.name)
    if match is None:
        raise ValueError(f"Lens filename must end in _step<number>_jlens.pt; got {path.name!r}.")
    model_id = match.group("model_part").replace("EleutherAI_", "EleutherAI/", 1)
    return model_id, match.group("revision"), int(match.group("step"))


def normalized_log_auc(curve: np.ndarray) -> float:
    curve = np.asarray(curve, dtype=np.float64)
    return float(np.trapezoid(curve, LOG_K_VALUES) / (LOG_K_VALUES[-1] - LOG_K_VALUES[0]))


def mean_stderr(values) -> tuple[float, float]:
    values = np.asarray(list(values), dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))


def one_based_rank(logits: torch.Tensor, token_id: int) -> int:
    target_logit = logits[int(token_id)]
    return int((logits > target_logit).sum().item()) + 1


def surface_forms(task: Task, intermediate: str) -> list[str]:
    """Surface strings scored for one intermediate.

    A byte-level BPE represents "Brazil" and " Brazil" as different tokens;
    they are the same word, so both are scored and the minimum rank is taken.
    """
    forms: list[str] = []
    for synonym in task.expand_synonyms(intermediate):
        forms += [synonym, " " + synonym]
    return list(dict.fromkeys(forms))


def unembed_stable(lens_model, residual: torch.Tensor) -> torch.Tensor:
    """Final norm + unembedding, with the normalization done in float32.

    This is the readout path behind every reported number. The explicit fp32
    norm keeps the transported residual ``J_l @ h`` safe even against a
    half-precision head: that product sums over d_model and can exceed fp16's
    65504 ceiling, and inf -> LayerNorm -> NaN logits would turn every rank
    into a silent 1. Post-norm values are O(1), so handing them to the head's
    dtype afterwards is safe; W_U is never upcast.
    """
    head = lens_model._lm_head
    norm = lens_model._final_norm
    x = residual.to(device=head.weight.device, dtype=torch.float32)

    mean = x.mean(dim=-1, keepdim=True)
    variance = x.var(dim=-1, unbiased=False, keepdim=True)
    x = (x - mean) * torch.rsqrt(variance + float(getattr(norm, "eps", 1e-5)))
    if getattr(norm, "weight", None) is not None:
        x = x * norm.weight.float()
    if getattr(norm, "bias", None) is not None:
        x = x + norm.bias.float()
    return head(x.to(head.weight.dtype)).float()


def load_eval_items(task: Task) -> tuple[Path, list[dict], str]:
    eval_path = DATA_ROOT / f"{task.slug}.json"
    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    return eval_path, payload["items"], sha256_file(eval_path)


def encode_with_offsets(tokenizer, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    encoded = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, truncation=False)
    input_ids = encoded.input_ids[0].tolist()
    offsets = [tuple(x) for x in encoded.offset_mapping[0].tolist()]
    return input_ids, offsets


def prompt_final_prepare(item_index: int, item: dict, tokenizer) -> dict:
    """Readout at the final prompt token; no answer target."""
    input_ids, _ = encode_with_offsets(tokenizer, item["prompt"])
    readout_position = len(input_ids) - 1
    return {
        "item_index": item_index,
        "name": item["name"],
        "input_ids": input_ids,
        "readout_position": readout_position,
        "answer_position": readout_position,
        "answer_word": None,
        "exclude_last_from_ablation": False,
        "intermediates": item["intermediates"],
    }


def target_boundary_prepare(item_index: int, item: dict, tokenizer) -> dict:
    """Readout at the token immediately preceding ``target``.

    The prompt and target are concatenated and offset-aligned rather than
    tokenized apart, so a trailing space in the prompt merges into the
    target's leading-space token instead of becoming a lone " " token.
    """
    target = item["target"]
    evaluation_text = item["prompt"] + target
    input_ids, offsets = encode_with_offsets(tokenizer, evaluation_text)
    boundary = len(item["prompt"])
    target_first = next(index for index, (start, end) in enumerate(offsets) if end > boundary)
    readout_position = target_first - 1
    return {
        "item_index": item_index,
        "name": item["name"],
        "input_ids": input_ids,
        "readout_position": readout_position,
        "answer_position": readout_position,
        "answer_word": target,
        "exclude_last_from_ablation": False,
        "intermediates": item["intermediates"],
    }


def build_tokenization(task: Task, tokenizer, items: list[dict]) -> tuple[dict, pd.DataFrame]:
    """Token ids for every surface form; single-token forms enter the paper metric."""
    unique = sorted({key for item in items for key in item["intermediates"]})
    forms_by_key: dict[str, list[dict]] = {}
    rows = []
    for key in unique:
        forms_by_key[key] = []
        for surface in surface_forms(task, key):
            token_ids = tokenizer.encode(surface, add_special_tokens=False)
            row = {
                "intermediate": key,
                "surface": surface,
                "token_ids": token_ids,
                "n_tokens": len(token_ids),
                "included_in_paper_metric": len(token_ids) == 1,
            }
            forms_by_key[key].append(row)
            rows.append(row)
    coverage = []
    for key, forms in forms_by_key.items():
        single = [f for f in forms if f["included_in_paper_metric"]]
        coverage.append({
            "intermediate": key,
            "scorable": bool(single),
            "n_single_token_forms": len(single),
            "single_token_surfaces": ", ".join(f["surface"] for f in single),
        })
    return forms_by_key, pd.DataFrame(coverage)


def paper_layers(all_layers: list[int]) -> list[int]:
    if len(all_layers) <= PAPER_LAYER_COUNT:
        return list(all_layers)
    indices = np.rint(np.linspace(0, len(all_layers) - 1, PAPER_LAYER_COUNT)).astype(int)
    return [all_layers[i] for i in indices]


def compute_metrics(
    items: list[dict],
    target_scores: pd.DataFrame,
    layer_sets: dict[str, list[int]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible = target_scores[target_scores["included_in_paper_metric"]].copy()
    base = pd.DataFrame([
        {"item_index": idx, "name": item["name"], "intermediate": key}
        for idx, item in enumerate(items)
        for key in item["intermediates"]
    ])

    item_rows = []
    summary_rows = []
    for method in METHODS:
        for layer_set_name, layers in layer_sets.items():
            candidate = eligible[(eligible["method"] == method) & (eligible["layer"].isin(layers))]
            if len(candidate):
                best_idx = candidate.groupby(["item_index", "intermediate"], sort=False)["rank"].idxmin()
                best = candidate.loc[best_idx, ["item_index", "intermediate", "rank"]]
            else:
                best = pd.DataFrame(columns=["item_index", "intermediate", "rank"])
            merged = base.merge(best, on=["item_index", "intermediate"], how="left")

            per_item = []
            for (item_index, name), group in merged.groupby(["item_index", "name"], sort=False):
                ranks = group["rank"].to_numpy(dtype=float)
                scorable = ranks[np.isfinite(ranks)]
                curve = (
                    (scorable[:, None] <= K_VALUES[None, :]).mean(axis=0)
                    if len(scorable)
                    else np.full(len(K_VALUES), np.nan)
                )
                per_item.append({
                    "method": method,
                    "layer_set": layer_set_name,
                    "item_index": int(item_index),
                    "name": name,
                    "n_intermediates": len(ranks),
                    "n_scorable": len(scorable),
                    "min_rank_best": float(scorable.min()) if len(scorable) else float("nan"),
                    "mrr": float(np.mean(1.0 / scorable)) if len(scorable) else float("nan"),
                    "eligible_auc": (
                        normalized_log_auc(curve) if np.isfinite(curve).all() else float("nan")
                    ),
                    **{f"pass_at_{int(k)}": float(v) for k, v in zip(K_VALUES, curve)},
                })
            item_df = pd.DataFrame(per_item)
            item_rows.append(item_df)

            scored = item_df[item_df["n_scorable"] > 0]
            auc_mean, auc_stderr = mean_stderr(scored["eligible_auc"])
            macro_curve = np.nanmean(
                scored[[f"pass_at_{int(k)}" for k in K_VALUES]].to_numpy(dtype=float), axis=0
            )
            summary_rows.append({
                "method": method,
                "layer_set": layer_set_name,
                "n_items": len(item_df),
                "n_items_scored": len(scored),
                "normalized_log_auc_k1_100": normalized_log_auc(macro_curve),
                "auc_item_standard_error": auc_stderr,
                "auc_item_mean": auc_mean,
                "mean_reciprocal_rank": float(scored["mrr"].mean()),
                **{f"pass_at_{int(k)}": float(v) for k, v in zip(K_VALUES, macro_curve)},
            })

    item_metrics = pd.concat(item_rows, ignore_index=True) if item_rows else pd.DataFrame()
    return item_metrics, pd.DataFrame(summary_rows)


def compute_layer_profile(target_scores: pd.DataFrame, n_layers: int) -> pd.DataFrame:
    """Per-layer recovery on depth reindexed to [0, 100], as the paper reports."""
    eligible = target_scores[target_scores["included_in_paper_metric"]]
    if not len(eligible):
        return pd.DataFrame()
    best = eligible.groupby(
        ["method", "layer", "item_index", "intermediate"], as_index=False
    )["rank"].min()
    best["layer_pct"] = 100.0 * best["layer"] / max(n_layers - 1, 1)
    return (
        best.assign(
            hit_at_1=lambda d: (d["rank"] <= 1).astype(float),
            hit_at_10=lambda d: (d["rank"] <= 10).astype(float),
            hit_at_100=lambda d: (d["rank"] <= 100).astype(float),
            reciprocal_rank=lambda d: 1.0 / d["rank"],
            log10_rank=lambda d: np.log10(d["rank"]),
        )
        .groupby(["method", "layer", "layer_pct"], as_index=False)
        .agg(
            pass_at_1=("hit_at_1", "mean"),
            pass_at_10=("hit_at_10", "mean"),
            pass_at_100=("hit_at_100", "mean"),
            mean_reciprocal_rank=("reciprocal_rank", "mean"),
            median_log10_rank=("log10_rank", "median"),
            n_scored=("rank", "size"),
        )
    )


def best_rank_and_logprob(
    task: Task, forms_by_key: dict, tokenizer, logits: torch.Tensor, word: str
) -> tuple[float, float]:
    """Best (rank, logprob) over the single-token surfaces of ``word``."""
    if not word:
        return float("nan"), float("nan")
    if word in forms_by_key:
        token_ids = [
            form["token_ids"][0] for form in forms_by_key[word]
            if form["included_in_paper_metric"]
        ]
    else:
        token_ids = []
        for surface in surface_forms(task, word):
            ids = tokenizer.encode(surface, add_special_tokens=False)
            if len(ids) == 1:
                token_ids.append(ids[0])
    if not token_ids:
        return float("nan"), float("nan")
    log_probs = logits.log_softmax(dim=-1)
    best_rank, best_logprob = float("inf"), float("nan")
    for token_id in token_ids:
        rank = one_based_rank(logits, token_id)
        if rank < best_rank:
            best_rank, best_logprob = float(rank), float(log_probs[token_id])
    return best_rank, best_logprob


@torch.no_grad()
def evaluate_lens(
    task: Task,
    lens_path: Path,
    items: list[dict],
    dataset_sha256: str,
    device: torch.device,
) -> tuple[dict, pd.DataFrame]:
    model_id, revision, checkpoint_step = identity_from_lens_filename(lens_path)
    lens_sha256 = sha256_file(lens_path)
    lens = load_lens(lens_path)
    hf_model, tokenizer = load_model(model_id, revision, device)
    lens_model = jlens.from_hf(hf_model, tokenizer, layout=pythia_layout(), compile=False)

    all_layers = list(lens.source_layers)
    layer_sets = {"all_fitted": all_layers, "paper_25": paper_layers(all_layers)}
    forms_by_key, tokenization_coverage = build_tokenization(task, tokenizer, items)
    audits = [task.prepare_item(i, item, tokenizer) for i, item in enumerate(items)]

    # One forward per item; keep the readout-position residual at every fitted
    # layer, plus the final-layer residual at the answer position (the model's
    # own next-token distribution, i.e. the capability control).
    residual_lists = {layer: [] for layer in all_layers}
    answer_residuals = []
    final_layer = lens_model.n_layers - 1
    record_layers = sorted(set(all_layers) | {final_layer})
    for audit in audits:
        input_ids = torch.tensor(
            [audit["input_ids"]], dtype=torch.long, device=lens_model.input_device
        )
        with ActivationRecorder(lens_model.layers, at=record_layers) as recorder:
            lens_model.forward(input_ids)
        position = audit["readout_position"]
        for layer in all_layers:
            residual_lists[layer].append(
                recorder.activations[layer][0, position].detach().to(device="cpu", dtype=torch.float32)
            )
        answer_residuals.append(
            recorder.activations[final_layer][0, audit["answer_position"]]
            .detach().to(device="cpu", dtype=torch.float32)
        )
        del recorder, input_ids

    readout_residuals = {layer: torch.stack(values) for layer, values in residual_lists.items()}

    score_rows = []
    for method in METHODS:
        use_jacobian = method == "jacobian"
        for layer in all_layers:
            jacobian = lens.jacobians[layer].to(device, dtype=torch.float32) if use_jacobian else None
            for batch_start in range(0, len(items), SCORE_BATCH_SIZE):
                batch_end = min(batch_start + SCORE_BATCH_SIZE, len(items))
                residual = readout_residuals[layer][batch_start:batch_end].to(device, dtype=torch.float32)
                if use_jacobian:
                    residual = residual @ jacobian.T
                logits = unembed_stable(lens_model, residual).cpu()
                log_z = torch.logsumexp(logits, dim=-1)
                for local_index in range(batch_end - batch_start):
                    item_index = batch_start + local_index
                    item_logits = logits[local_index]
                    item_log_z = float(log_z[local_index])
                    for key in items[item_index]["intermediates"]:
                        for form in forms_by_key[key]:
                            if not form["token_ids"]:
                                continue
                            token_id = form["token_ids"][0]
                            logit = float(item_logits[token_id])
                            score_rows.append({
                                "item_index": item_index,
                                "name": items[item_index]["name"],
                                "method": method,
                                "layer": layer,
                                "intermediate": key,
                                "surface": form["surface"],
                                "included_in_paper_metric": form["included_in_paper_metric"],
                                "rank": one_based_rank(item_logits, token_id),
                                "logit": logit,
                                "logprob": logit - item_log_z,
                            })
            if use_jacobian:
                del jacobian
            if device.type == "cuda":
                torch.cuda.empty_cache()

    target_scores = pd.DataFrame(score_rows)
    item_metrics, run_summary = compute_metrics(items, target_scores, layer_sets)
    layer_profile = compute_layer_profile(target_scores, lens_model.n_layers)

    # Capability control: can this checkpoint produce the task's answer at all?
    control_rows = []
    for batch_start in range(0, len(items), SCORE_BATCH_SIZE):
        batch_end = min(batch_start + SCORE_BATCH_SIZE, len(items))
        residual = torch.stack(answer_residuals[batch_start:batch_end]).to(device, dtype=torch.float32)
        logits = unembed_stable(lens_model, residual).cpu()
        for local_index in range(batch_end - batch_start):
            item_index = batch_start + local_index
            item_logits = logits[local_index]
            audit = audits[item_index]
            rank, logprob = best_rank_and_logprob(
                task, forms_by_key, tokenizer, item_logits, audit["answer_word"] or ""
            )
            control_rows.append({
                "item_index": item_index,
                "name": items[item_index]["name"],
                "readout_position": audit["readout_position"],
                "readout_token": tokenizer.decode([audit["input_ids"][audit["readout_position"]]]),
                "answer_position": audit["answer_position"],
                "answer_word": audit["answer_word"] or "",
                "model_top1": tokenizer.decode([int(item_logits.argmax())]),
                "answer_rank": rank,
                "answer_logprob": logprob,
                "answer_is_top1": bool(rank == 1),
            })
    capability_control = pd.DataFrame(control_rows)
    # Capability averages run over scorable items only: a NaN rank means the
    # item had no target or no single-token surface -- "we could not ask",
    # not "the model got it wrong".
    scorable = capability_control[capability_control["answer_rank"].notna()]
    finite_rank = scorable["answer_rank"]

    n_lens_fit_prompts = int(lens.n_prompts)
    del hf_model, lens_model, lens
    if device.type == "cuda":
        torch.cuda.empty_cache()

    payload = {
        "evaluation": task.name,
        "evaluation_slug": task.slug,
        "readout_mode": task.readout_mode,
        "lens_filename": lens_path.name,
        "lens_sha256": lens_sha256,
        "model_id": model_id,
        "revision": revision,
        "checkpoint_step": checkpoint_step,
        "parameter_count": PARAM_COUNTS.get(model_id),
        "dtype": "float32",
        "device": str(device),
        "tokens_seen": checkpoint_step * TOKENS_PER_STEP,
        "dataset_sha256": dataset_sha256,
        "n_items": len(items),
        "n_unique_intermediates": len(tokenization_coverage),
        "n_scorable_intermediates": int(tokenization_coverage["scorable"].sum()),
        "tokenization_coverage": float(tokenization_coverage["scorable"].mean()),
        "fitted_layers": all_layers,
        "paper_25_layers": layer_sets["paper_25"],
        "n_model_layers": int(len(all_layers) + 1),
        "methods": METHODS,
        "paper_k_values": K_VALUES.tolist(),
        "n_lens_fit_prompts": n_lens_fit_prompts,
        "n_capability_scorable": int(len(scorable)),
        "capability_answer_top1_rate": (
            float(scorable["answer_is_top1"].mean()) if len(scorable) else None
        ),
        "capability_answer_median_rank": (
            float(finite_rank.median()) if len(finite_rank) else None
        ),
        "run_summary": run_summary.to_dict(orient="records"),
        "item_metrics": item_metrics.to_dict(orient="records"),
        "layer_profile": layer_profile.to_dict(orient="records"),
        "capability_control": capability_control.to_dict(orient="records"),
        "tokenization_coverage_by_intermediate": tokenization_coverage.to_dict(orient="records"),
    }
    return payload, target_scores


def output_path_for_lens(lens_path: Path, output_dir: Path, eval_name: str, suffix: str = ".json") -> Path:
    match = re.fullmatch(r"(?P<model_part>.+)_(?P<revision>step\d+)_jlens\.pt", lens_path.name)
    if match is None:
        raise ValueError(f"Lens filename must end in _step<number>_jlens.pt; got {lens_path.name!r}.")
    stem = lens_path.name.replace("_jlens.pt", f"_{eval_name}{suffix}")
    return output_dir / match.group("model_part") / match.group("revision") / stem


def rebuild_summary(output_dir: Path, eval_name: str) -> Path | None:
    """Rebuild emergence_summary_<eval>.csv by scanning nested result JSONs."""
    rows: list[dict] = []
    for path in sorted(output_dir.resolve().rglob(f"*_{eval_name}.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload["run_summary"]:
            rows.append({
                "evaluation": payload["evaluation"],
                "lens_filename": payload["lens_filename"],
                **row,
                "model_id": payload["model_id"],
                "revision": payload["revision"],
                "checkpoint_step": payload["checkpoint_step"],
                "parameter_count": payload["parameter_count"],
                "tokens_seen": payload["tokens_seen"],
                "capability_answer_top1_rate": payload["capability_answer_top1_rate"],
                "capability_answer_median_rank": payload["capability_answer_median_rank"],
                "n_capability_scorable": payload["n_capability_scorable"],
                "n_scorable_intermediates": payload["n_scorable_intermediates"],
                "tokenization_coverage": payload["tokenization_coverage"],
                "lens_sha256": payload["lens_sha256"],
                "n_lens_fit_prompts": payload["n_lens_fit_prompts"],
                "dtype": payload["dtype"],
                "device": payload["device"],
            })
    if not rows:
        print(f"No {eval_name} results found under {output_dir}")
        return None
    frame = pd.DataFrame(rows).sort_values(
        ["method", "layer_set", "parameter_count", "checkpoint_step"], na_position="last"
    )
    summary_path = output_dir / f"emergence_summary_{eval_name}.csv"
    frame.to_csv(summary_path, index=False)
    print(f"{len(frame)} summary rows -> {summary_path}")
    return summary_path


def run(task: Task) -> int:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eval_path, items, dataset_sha256 = load_eval_items(task)
    lenses = discover_lenses(FITS_ROOT)

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Evaluation: {task.name} ({len(items)} items from {eval_path.name})")
    print(f"Lenses: {len(lenses)}; output: {RESULTS_ROOT} (<model>/<step>/*_{task.name}.json)")

    for index, lens_path in enumerate(lenses, start=1):
        out_path = output_path_for_lens(lens_path, RESULTS_ROOT, task.name)
        print(f"[{index}/{len(lenses)}] {lens_path.name}")
        started = time.perf_counter()
        payload, target_scores = evaluate_lens(task, lens_path, items, dataset_sha256, device)
        payload["elapsed_seconds"] = round(time.perf_counter() - started, 2)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        scores_path = output_path_for_lens(
            lens_path, RESULTS_ROOT, task.name, suffix="_layer_scores.csv.gz"
        )
        target_scores.to_csv(scores_path, index=False, compression="gzip")
        payload["layer_scores_file"] = scores_path.name
        out_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=json_default),
            encoding="utf-8",
        )
        headline = next(
            row for row in payload["run_summary"]
            if row["method"] == "jacobian" and row["layer_set"] == "all_fitted"
        )
        print(
            f"  -> {out_path.name} ({payload['elapsed_seconds']}s)"
            f" AUC={headline['normalized_log_auc_k1_100']:.4f}"
            f" pass@1={headline['pass_at_1']:.3f}"
            f" pass@10={headline['pass_at_10']:.3f}"
        )

    rebuild_summary(RESULTS_ROOT, task.name)
    return 0
