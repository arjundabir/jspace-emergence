from __future__ import annotations

import pandas as pd
import torch

from experiments import common
from experiments.common import LensRun

EXPERIMENT_NAME = "dual-task"
K_VALUES = [1, 5, 10]
REACH_K = 5


def expr_of(pair: dict) -> str:
    return f"{pair['base']}^{pair['exp']} - {pair['sub']}"


def answer_of(pair: dict) -> int:
    return pair["base"] ** pair["exp"] - pair["sub"]


def concept_task_ids(tokenizer, pair: dict) -> list[int]:
    ids: list[int] = []
    for member in pair["concept_words"]:
        ids += common.word_variant_ids(tokenizer, member)
    return sorted(set(ids))


def instruction_for(condition: str, pair: dict, partner: dict | None) -> str:
    """The covert-task instruction sentence for one trial."""
    if condition in ("concept", "a_only"):
        return f"Concentrate on {pair['concept']} while you write the sentence."
    if condition == "b_only":
        return f"Concentrate on {partner['concept']} while you write the sentence."
    if condition == "math":
        return f"Try to focus on evaluating {expr_of(pair)} while you write the sentence."
    if condition == "concept_first":
        return (
            f"Concentrate on {pair['concept']} and try to focus on evaluating "
            f"{expr_of(pair)} while you write the sentence."
        )
    if condition == "math_first":
        return (
            f"Try to focus on evaluating {expr_of(pair)} and concentrate on "
            f"{pair['concept']} while you write the sentence."
        )
    if condition == "a_first":
        return (
            f"Concentrate on {pair['concept']} and {partner['concept']} "
            "while you write the sentence."
        )
    if condition == "b_first":
        return (
            f"Concentrate on {partner['concept']} and {pair['concept']} "
            "while you write the sentence."
        )
    raise ValueError(f"Unknown condition {condition!r}")


@torch.no_grad()
def best_ranks_for_tasks(
    run: LensRun, prefix: str, response: str, task_ids: dict[str, list[int]]
) -> dict[str, dict[str, float]]:
    """``{method: {task: best rank over (band layer, response position)}}``."""
    input_ids, positions = common.teacher_forced_span(run.tokenizer, prefix, response)
    residuals = common.capture_band_residuals(run, input_ids, positions)
    best: dict[str, dict[str, float]] = {
        method: {task: float("inf") for task in task_ids} for method in common.METHODS
    }
    for method in common.METHODS:
        for layer in run.band:
            logits = run.lens_logits(residuals[layer], layer, method)
            for task, ids in task_ids.items():
                layer_min = common.ranks_of(logits, ids).min().item()
                best[method][task] = min(best[method][task], float(layer_min))
            del logits
    del residuals
    return best


@torch.no_grad()
def run_one(run: LensRun) -> tuple[dict, dict[str, pd.DataFrame]]:
    data_path, data, data_sha256 = common.load_experiment_data(EXPERIMENT_NAME)
    carrier = data["carrier_sentence"]
    pairs = {pair["key"]: pair for pair in data["pairs"]}

    rows = []

    def run_condition(
        arm: str,
        condition: str,
        pair_key: str,
        partner_key: str,
        instruction: str,
        task_ids: dict[str, list[int]],
        task_labels: dict[str, str],
        is_dual: bool,
        single_condition_of: dict[str, str],
    ) -> None:
        # Tokenless tasks are excluded from the denominator, not failures.
        scorable = {task: ids for task, ids in task_ids.items() if ids}
        if not scorable:
            return
        prefix, response = common.chat_prompt(carrier, instruction)
        best = best_ranks_for_tasks(run, prefix, response, scorable)
        for method in common.METHODS:
            for task, rank in best[method].items():
                rows.append({
                    "method": method,
                    "arm": arm,
                    "condition": condition,
                    "pair_key": pair_key,
                    "partner_key": partner_key,
                    "task": task,
                    "task_label": task_labels[task],
                    "instructed": is_dual or condition == single_condition_of[task],
                    "dual": is_dual,
                    "n_tracked_tokens": len(scorable[task]),
                    "best_rank": rank,
                    **{f"hit{k}": rank <= k for k in K_VALUES},
                    "reach": rank <= REACH_K,
                })

    for pair in data["pairs"]:
        task_ids = {
            "concept": concept_task_ids(run.tokenizer, pair),
            "math": common.number_target_ids(run.tokenizer, answer_of(pair)),
        }
        task_labels = {"concept": pair["concept"], "math": expr_of(pair)}
        for condition in data["concept_math_conditions"]:
            run_condition(
                "concept_math",
                condition,
                pair["key"],
                "",
                instruction_for(condition, pair, None),
                task_ids,
                task_labels,
                is_dual=condition in ("concept_first", "math_first"),
                single_condition_of={"concept": "concept", "math": "math"},
            )
        if run.device.type == "cuda":
            torch.cuda.empty_cache()

    for key_a, key_b in data["concept_pairs"]:
        pair_a, pair_b = pairs[key_a], pairs[key_b]
        task_ids = {
            "a": concept_task_ids(run.tokenizer, pair_a),
            "b": concept_task_ids(run.tokenizer, pair_b),
        }
        task_labels = {"a": pair_a["concept"], "b": pair_b["concept"]}
        for condition in data["concept_concept_conditions"]:
            run_condition(
                "concept_concept",
                condition,
                key_a,
                key_b,
                instruction_for(condition, pair_a, pair_b),
                task_ids,
                task_labels,
                is_dual=condition in ("a_first", "b_first"),
                single_condition_of={"a": "a_only", "b": "b_only"},
            )
        if run.device.type == "cuda":
            torch.cuda.empty_cache()

    scores = pd.DataFrame(rows)

    summary_rows = []
    for (method, arm, condition, task), group in scores.groupby(
        ["method", "arm", "condition", "task"], sort=False
    ):
        reach_rate, reach_stderr = common.mean_stderr(group["reach"].astype(float))
        entry = {
            "method": method,
            "arm": arm,
            "condition": condition,
            "task": task,
            "instructed": bool(group["instructed"].iloc[0]),
            "n_trials": len(group),
            "reach_rate": reach_rate,
            "reach_stderr": reach_stderr,
            "median_best_rank": float(group["best_rank"].median()),
            "mean_reciprocal_best_rank": float((1.0 / group["best_rank"]).mean()),
        }
        for k in K_VALUES:
            entry[f"hit{k}_rate"] = float(group[f"hit{k}"].mean())
        summary_rows.append(entry)

    def reach(method: str, arm: str, task: str, conditions: list[str]) -> float:
        subset = scores[
            (scores["method"] == method)
            & (scores["arm"] == arm)
            & (scores["task"] == task)
            & (scores["condition"].isin(conditions))
        ]
        return float(subset["reach"].mean()) if len(subset) else float("nan")

    interference = []
    for method in common.METHODS:
        for arm, single_of, duals in (
            ("concept_math", {"concept": ["concept"], "math": ["math"]},
             ["concept_first", "math_first"]),
            ("concept_concept", {"a": ["a_only"], "b": ["b_only"]},
             ["a_first", "b_first"]),
        ):
            for task, single_conditions in single_of.items():
                single = reach(method, arm, task, single_conditions)
                dual = reach(method, arm, task, duals)
                interference.append({
                    "method": method,
                    "arm": arm,
                    "task": task,
                    "single_reach": single,
                    "dual_reach": dual,
                    "interference": single - dual,
                })

    jacobian = [row for row in interference if row["method"] == "jacobian"]
    payload = {
        "dataset_sha256": data_sha256,
        "carrier_sentence": carrier,
        "reach_k": REACH_K,
        "k_values": K_VALUES,
        "run_summary": summary_rows,
        "interference": interference,
        "headline": " ".join(
            f"{row['arm']}/{row['task']}: single={row['single_reach']:.2f} "
            f"dual={row['dual_reach']:.2f}"
            for row in jacobian
            if row["arm"] == "concept_math"
        ),
    }
    return payload, {"_trial_scores.csv.gz": scores}


if __name__ == "__main__":
    raise SystemExit(common.run_experiment(EXPERIMENT_NAME, run_one))
