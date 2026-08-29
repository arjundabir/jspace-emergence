from __future__ import annotations

import random

import pandas as pd
import torch

from experiments import common
from experiments.common import LensRun

EXPERIMENT_NAME = "directed-modulation"
DATA_SLUG = "directed-modulation"
K_VALUES = [1, 5, 10]
FAMILIES = ["category", "math"]
NONE_PHRASING = {"name": "none", "group": "none", "text": None}
TRIALS_PER_CELL = 6


def target_tokens(tokenizer, family: str, target: dict) -> tuple[str, list[int]]:
    if family == "category":
        ids: list[int] = []
        for member in target["members"]:
            ids += common.word_variant_ids(tokenizer, member)
        return target["name"], sorted(set(ids))
    return target["expr"], common.number_target_ids(tokenizer, target["answer"])


def fill_x(family: str, target: dict) -> str:
    return target["name"] if family == "category" else f"evaluating {target['expr']}"


@torch.no_grad()
def run_one(run: LensRun) -> tuple[dict, dict[str, pd.DataFrame]]:
    data_path, data, data_sha256 = common.load_experiment_data(DATA_SLUG)
    phrasings = [NONE_PHRASING] + data["phrasings"]
    group_kind = {**data["group_kind"], "none": "baseline"}
    carriers = data["carrier_sentences"]
    targets_by_family = {
        "category": data["topic_categories"],
        "math": data["math_problems"],
    }

    # One shared (carrier, target) draw per (family, trial): every phrasing
    # cell — including the no-instruction baseline — scores identical items,
    # so the focus/mention/suppress/baseline contrasts are item-matched.
    trial_specs: dict[str, list[tuple[int, dict]]] = {}
    for family_index, family in enumerate(FAMILIES):
        rng = random.Random(common.RANDOM_SEED * 1_000_003 + family_index)
        targets = targets_by_family[family]
        trial_specs[family] = [
            (rng.randrange(len(carriers)), rng.choice(targets))
            for _ in range(TRIALS_PER_CELL)
        ]

    rows = []
    for phrasing in phrasings:
        for family in FAMILIES:
            for trial, (carrier_index, target) in enumerate(trial_specs[family]):
                carrier = carriers[carrier_index]
                display, tracked_ids = target_tokens(run.tokenizer, family, target)
                if not tracked_ids:
                    continue  # excluded from the denominator, not a failure

                instruction = (
                    None
                    if phrasing["text"] is None
                    else phrasing["text"].format(x=fill_x(family, target))
                )
                prefix, response = common.chat_prompt(carrier, instruction)
                input_ids, positions = common.teacher_forced_span(
                    run.tokenizer, prefix, response
                )
                residuals = common.capture_band_residuals(run, input_ids, positions)

                for method in common.METHODS:
                    best_rank = float("inf")
                    for layer in run.band:
                        logits = run.lens_logits(residuals[layer], layer, method)
                        layer_min = common.ranks_of(logits, tracked_ids).min().item()
                        best_rank = min(best_rank, float(layer_min))
                        del logits
                    rows.append({
                        "method": method,
                        "phrasing": phrasing["name"],
                        "group": phrasing["group"],
                        "kind": group_kind[phrasing["group"]],
                        "family": family,
                        "trial": trial,
                        "carrier_index": carrier_index,
                        "target": display,
                        "tier": target.get("tier", ""),
                        "n_tracked_tokens": len(tracked_ids),
                        "n_response_positions": len(positions),
                        "best_rank": best_rank,
                        **{f"hit{k}": best_rank <= k for k in K_VALUES},
                    })
                del residuals
        if run.device.type == "cuda":
            torch.cuda.empty_cache()

    scores = pd.DataFrame(rows)

    summary_rows = []
    for (method, kind, family), group in scores.groupby(
        ["method", "kind", "family"], sort=False
    ):
        entry = {
            "method": method,
            "kind": kind,
            "family": family,
            "n_trials": len(group),
            "mean_reciprocal_best_rank": float((1.0 / group["best_rank"]).mean()),
            "median_best_rank": float(group["best_rank"].median()),
        }
        for k in K_VALUES:
            rate, stderr = common.mean_stderr(group[f"hit{k}"].astype(float))
            entry[f"hit{k}_rate"] = rate
            entry[f"hit{k}_stderr"] = stderr
        summary_rows.append(entry)

    phrasing_summary = [
        {
            "method": method,
            "phrasing": phrasing,
            "kind": group["kind"].iloc[0],
            "family": family,
            "n_trials": len(group),
            "hit1_rate": float(group["hit1"].mean()),
            "hit5_rate": float(group["hit5"].mean()),
            "median_best_rank": float(group["best_rank"].median()),
        }
        for (method, phrasing, family), group in scores.groupby(
            ["method", "phrasing", "family"], sort=False
        )
    ]

    def rate(kind: str) -> float:
        match = [
            row["hit1_rate"]
            for row in summary_rows
            if row["method"] == "jacobian" and row["kind"] == kind
        ]
        return float(pd.Series(match).mean()) if match else float("nan")

    payload = {
        "dataset_sha256": data_sha256,
        "families": FAMILIES,
        "k_values": K_VALUES,
        "trials_per_cell": TRIALS_PER_CELL,
        "line_width_family": "omitted (prose corpus not released with the dataset)",
        "run_summary": summary_rows,
        "phrasing_summary": phrasing_summary,
        "headline": (
            f"hit@1 focus={rate('focus'):.3f} control={rate('control'):.3f} "
            f"suppress={rate('suppress'):.3f} baseline={rate('baseline'):.3f}"
        ),
    }
    return payload, {"_trial_scores.csv.gz": scores}


if __name__ == "__main__":
    raise SystemExit(common.run_experiment(EXPERIMENT_NAME, run_one))
