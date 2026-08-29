from __future__ import annotations

import gc
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
import torch

import jlens

from lens.load_lens import FITS_ROOT, load_lens, lens_path_for, pythia_layout
from lens.load_models import CHECKPOINT_STEPS, REPO_ROOT, load_model

DATA_ROOT = REPO_ROOT / "jacobian-lens" / "data" / "evaluations"
RESULTS_ROOT = REPO_ROOT / "results" / "ablations"

# Workspace band as a fraction of depth, fixed across models and checkpoints
# so the emergence claim compares the same band at every step. Located on
# Pythia by the excess-kurtosis minimum of the fitted lens, which sits inside
# (0.35, 0.90) at the last three checkpoints of every sizeable model.
BAND_FRACTION = (0.35, 0.90)


def band_layers(all_layers: list[int], n_layers: int) -> list[int]:
    last = max(n_layers - 1, 1)
    low, high = BAND_FRACTION
    inside = [layer for layer in all_layers if low <= layer / last <= high]
    return inside or list(all_layers)


def surface_forms(word: str) -> list[str]:
    return list(dict.fromkeys([word, " " + word]))


def first_single_token_id(tokenizer, word: str) -> tuple[int, str]:
    for surface in surface_forms(word):
        ids = tokenizer.encode(surface, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0], surface
    raise ValueError(f"No single-token form for {word!r}")


def try_single_token_id(tokenizer, word: str) -> tuple[int, str] | None:
    try:
        return first_single_token_id(tokenizer, word)
    except ValueError:
        return None


def one_based_rank(logits: torch.Tensor, token_id: int) -> int:
    return int((logits > logits[int(token_id)]).sum().item()) + 1


def best_rank_logprob(tokenizer, logits: torch.Tensor, word: str):
    log_probs = logits.float().log_softmax(dim=-1)
    best = (float("inf"), float("nan"), None)
    for surface in surface_forms(word):
        ids = tokenizer.encode(surface, add_special_tokens=False)
        if len(ids) != 1:
            continue
        tid = ids[0]
        rank = one_based_rank(logits, tid)
        if rank < best[0]:
            best = (float(rank), float(log_probs[tid]), surface)
    return best


def teacher_forced_mean_logprob(
    logits: torch.Tensor, input_ids_list: list[int], target_positions: list[int]
) -> float:
    total = 0.0
    for pos in target_positions:
        log_probs = logits[pos - 1].float().log_softmax(dim=-1)
        total += float(log_probs[int(input_ids_list[pos])])
    return total / float(len(target_positions))


def lens_vector(lens, lens_model, layer: int, token_id: int, method: str) -> torch.Tensor:
    unembed_row = lens_model._lm_head.weight[token_id].detach().float().cpu()
    if method == "logit":
        vector = unembed_row.clone()
    else:
        vector = lens.jacobians[layer].float().t() @ unembed_row
    return vector / vector.norm().clamp_min(1e-8)


def make_ablation_hook(vector: torch.Tensor, positions: list[int]):
    def hook(module, inputs, output):
        if torch.is_tensor(output):
            hidden, rest = output, None
        else:
            hidden, rest = output[0], output[1:]
        unit = vector.to(hidden.device, torch.float32)
        hidden = hidden.clone()
        block = hidden[:, positions, :].float()
        block = block - (block @ unit).unsqueeze(-1) * unit
        hidden[:, positions, :] = block.to(hidden.dtype)
        return hidden if rest is None else (hidden, *rest)

    return hook


@torch.no_grad()
def forward_with_hooks(hf_model, lens_model, input_ids, hooks: dict):
    handles = [
        lens_model.layers[layer].register_forward_hook(hook)
        for layer, hook in hooks.items()
    ]
    logits = hf_model(input_ids=input_ids).logits[0].float()
    for handle in handles:
        handle.remove()
    return logits


def kl_divergence(clean_logits: torch.Tensor, other_logits: torch.Tensor) -> float:
    log_p = clean_logits.float().log_softmax(dim=-1)
    log_q = other_logits.float().log_softmax(dim=-1)
    return float((log_p.exp() * (log_p - log_q)).sum(dim=-1))


@dataclass(frozen=True)
class Example:
    input_ids_list: list[int]
    readout_position: int
    ablate_positions: list[int]
    direction_token_id: int
    intermediate: str
    target: str
    target_positions: list[int] | None = None  # whole-answer span, else None
    rank_scorable: bool = True


@dataclass(frozen=True)
class Task:
    name: str
    slug: str
    prepare_example: Callable[[dict, object], "Example | str"]  # str = skip reason
    whole_answer: bool = False


def prompt_final_prepare(item: dict, tokenizer) -> Example | str:
    concept = item["intermediates"][0]
    try:
        concept_id, _ = first_single_token_id(tokenizer, concept)
    except ValueError as exc:
        return str(exc)
    input_ids_list = tokenizer(
        item["prompt"], return_tensors="pt", truncation=False
    ).input_ids[0].tolist()
    return Example(
        input_ids_list=input_ids_list,
        readout_position=len(input_ids_list) - 1,
        ablate_positions=list(range(len(input_ids_list))),
        direction_token_id=concept_id,
        intermediate=concept,
        target=concept,
    )


def _boundary_encode(item: dict, tokenizer, target: str) -> tuple[list[int], int]:
    encoded = tokenizer(
        item["prompt"] + target,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=False,
    )
    input_ids_list = encoded.input_ids[0].tolist()
    offsets = [tuple(x) for x in encoded.offset_mapping[0].tolist()]
    boundary = len(item["prompt"])
    target_first = next(i for i, (start, end) in enumerate(offsets) if end > boundary)
    return input_ids_list, target_first


def target_boundary_prepare(item: dict, tokenizer) -> Example | str:
    target = item["target"]
    intermediate = item["intermediates"][0]
    try:
        intermediate_id, _ = first_single_token_id(tokenizer, intermediate)
        first_single_token_id(tokenizer, target)
    except ValueError as exc:
        return str(exc)
    input_ids_list, target_first = _boundary_encode(item, tokenizer, target)
    return Example(
        input_ids_list=input_ids_list,
        readout_position=target_first - 1,
        ablate_positions=list(range(len(input_ids_list))),
        direction_token_id=intermediate_id,
        intermediate=intermediate,
        target=target,
    )


def whole_answer_prepare(item: dict, tokenizer) -> Example | str:
    intermediate = item["intermediates"][0]
    target = item.get("target") or intermediate
    try:
        intermediate_id, _ = first_single_token_id(tokenizer, intermediate)
    except ValueError as exc:
        return str(exc)
    input_ids_list, target_first = _boundary_encode(item, tokenizer, target)
    return Example(
        input_ids_list=input_ids_list,
        readout_position=target_first - 1,
        ablate_positions=list(range(len(input_ids_list))),
        direction_token_id=intermediate_id,
        intermediate=intermediate,
        target=target,
        target_positions=list(range(target_first, len(input_ids_list))),
        rank_scorable=try_single_token_id(tokenizer, target) is not None,
    )


METHODS = ("jacobian", "logit")

SUMMARY_COLUMNS = [
    "training_step", "method", "total_examples", "usable_examples",
    "mean_kl", "median_kl", "mean_rank_change", "mean_logprob_change",
]

PER_EXAMPLE_COLUMNS = [
    "training_step", "method", "item_index", "name", "intermediate", "target",
    "answer_surface", "rank_clean", "rank_ablated", "logprob_clean",
    "logprob_ablated", "kl", "delta_rank", "delta_logprob",
]


def run_ablation(
    task: Task, model_id: str, step: int, lens_path: Path, device: torch.device
) -> tuple[list[dict], list[dict]]:
    revision = f"step{step}"
    lens = load_lens(lens_path)
    hf_model, tokenizer = load_model(model_id, revision, device)
    lens_model = jlens.from_hf(hf_model, tokenizer, layout=pythia_layout(), compile=False)

    all_items = json.loads((DATA_ROOT / f"{task.slug}.json").read_text(encoding="utf-8"))["items"]
    workspace = band_layers(list(lens.source_layers), lens_model.n_layers)

    rows: list[dict] = []
    n_skipped = 0
    for item_index, item in enumerate(all_items):
        example = task.prepare_example(item, tokenizer)
        if isinstance(example, str):
            n_skipped += 1
            continue

        input_ids = torch.tensor([example.input_ids_list], dtype=torch.long, device=device)
        with torch.no_grad():
            clean_logits = hf_model(input_ids=input_ids).logits[0].float()
        clean_readout = clean_logits[example.readout_position]
        if task.whole_answer:
            logprob_clean = teacher_forced_mean_logprob(
                clean_logits, example.input_ids_list, example.target_positions
            )
            if example.rank_scorable:
                rank_clean, _, answer_surface = best_rank_logprob(
                    tokenizer, clean_readout, example.target
                )
            else:
                rank_clean, answer_surface = float("nan"), ""
        else:
            rank_clean, logprob_clean, answer_surface = best_rank_logprob(
                tokenizer, clean_readout, example.target
            )

        for method in METHODS:
            hooks = {
                layer: make_ablation_hook(
                    lens_vector(lens, lens_model, layer, example.direction_token_id, method),
                    example.ablate_positions,
                )
                for layer in workspace
            }
            ablated_logits = forward_with_hooks(hf_model, lens_model, input_ids, hooks)
            ablated_readout = ablated_logits[example.readout_position]
            kl = kl_divergence(clean_readout, ablated_readout)

            if task.whole_answer:
                logprob_ablated = teacher_forced_mean_logprob(
                    ablated_logits, example.input_ids_list, example.target_positions
                )
                if example.rank_scorable:
                    rank_ablated, _, _ = best_rank_logprob(
                        tokenizer, ablated_readout, example.target
                    )
                else:
                    rank_ablated = float("nan")
            else:
                rank_ablated, logprob_ablated, _ = best_rank_logprob(
                    tokenizer, ablated_readout, example.target
                )

            rows.append({
                "training_step": int(step),
                "method": method,
                "item_index": item_index,
                "name": item["name"],
                "intermediate": example.intermediate,
                "target": example.target,
                "answer_surface": answer_surface,
                "rank_clean": rank_clean,
                "rank_ablated": rank_ablated,
                "logprob_clean": logprob_clean,
                "logprob_ablated": logprob_ablated,
                "kl": kl,
                "delta_rank": rank_ablated - rank_clean,
                "delta_logprob": logprob_ablated - logprob_clean,
            })

    del hf_model, lens_model, lens
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    per_example = pd.DataFrame(rows)
    summary_rows: list[dict] = []
    for method in METHODS:
        sub = per_example[per_example["method"] == method] if len(per_example) else per_example
        rank_rows = sub[sub["rank_clean"].notna()] if len(sub) else sub
        summary_rows.append({
            "training_step": int(step),
            "method": method,
            "total_examples": len(all_items),
            "usable_examples": int(len(sub)),
            "mean_kl": float(sub["kl"].mean()) if len(sub) else float("nan"),
            "median_kl": float(sub["kl"].median()) if len(sub) else float("nan"),
            "mean_rank_change": float(rank_rows["delta_rank"].mean()) if len(rank_rows) else float("nan"),
            "mean_logprob_change": float(sub["delta_logprob"].mean()) if len(sub) else float("nan"),
        })

    print(f"model:            {model_id} @ {revision}")
    print(f"workspace layers: {workspace}")
    print(f"usable examples:  {len(all_items) - n_skipped}/{len(all_items)}")
    for row in summary_rows:
        print(
            f"[{row['method']}] mean_kl={row['mean_kl']:.6f}  "
            f"median_kl={row['median_kl']:.6f}  "
            f"mean_rank_change={row['mean_rank_change']:.6f}  "
            f"mean_logprob_change={row['mean_logprob_change']:.6f}"
        )
    return summary_rows, rows


def run(task: Task) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for model_dir in sorted(FITS_ROOT.iterdir()):
        jobs = sorted(
            (int(re.search(r"_step(\d+)_jlens\.pt$", path.name).group(1)), path)
            for path in model_dir.glob("*_jlens.pt")
        )
        model_id = model_dir.name.replace("EleutherAI_", "EleutherAI/", 1)

        summary_rows: list[dict] = []
        per_example_rows: list[dict] = []
        for step, lens_path in jobs:
            print(f"\n=== {model_dir.name} step {step} ===")
            step_summary, step_rows = run_ablation(task, model_id, step, lens_path, device)
            summary_rows.extend(step_summary)
            per_example_rows.extend(step_rows)

        out_dir = RESULTS_ROOT / model_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_csv = out_dir / f"{task.name}_ablation_summary.csv"
        per_example_csv = out_dir / f"{task.name}_ablation_per_example.csv"
        summary = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
        summary = summary.sort_values(["training_step", "method"]).reset_index(drop=True)
        summary.to_csv(summary_csv, index=False)
        per_example = pd.DataFrame(per_example_rows, columns=PER_EXAMPLE_COLUMNS)
        per_example = per_example.sort_values(["training_step", "item_index"]).reset_index(drop=True)
        per_example.to_csv(per_example_csv, index=False)
        print(f"\nSaved summary:     {summary_csv}")
        print(f"Saved per-example: {per_example_csv} ({len(per_example)} rows)")
    return 0
