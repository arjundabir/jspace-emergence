"""Line-count selectivity experiment (paper §3.7, Figure 27).

Each passage is wrapped with ``textwrap.fill`` at its ``width``; the
ground-truth answer is the character count of the first wrapped line. Four
conditions put different demands on that count: **none** (no question,
prefill ``" The first line has"`` — automatic floor), **direct** (asks for
the count, same prefill), **letter** (asks for the first letter of the
spelled-out count — the count is needed internally but is not the answer
token), and **continue** (asks to continue the passage with the same
wrapping — the count is needed but never verbalized).

The lens is applied at every prompt position across the workspace band,
tracking the *expected* count (digits and English number word) and the
generic *any-number* set (every two-digit token and number word 10-99).
Selectivity is how much more strongly count information surfaces when the
task requires it. A behavioral check at the final position records the
expected answer's rank and probability.

    python -m experiments.selectivity_linecount
"""

from __future__ import annotations

import textwrap

import pandas as pd
import torch

from experiments import common
from experiments.common import LensRun

EXPERIMENT_NAME = "selectivity-linecount"
K_VALUES = [1, 5, 10]
PREFILL_CONDITIONS = ["none", "direct", "letter"]
CONDITIONS = PREFILL_CONDITIONS + ["continue"]


def any_number_ids(tokenizer) -> list[int]:
    """Single-token forms of every two-digit numeral and English number word
    10-99 — the dataset's generic number-token target set."""
    ids: list[int] = []
    for value in range(10, 100):
        for surface in (str(value), common.number_word(value)):
            ids += common.word_variant_ids(tokenizer, surface)
    return sorted(set(ids))


def build_prompt(condition: str, data: dict, wrapped: str) -> tuple[str, str]:
    """(prompt prefix ending at ``Assistant:``, teacher-forced prefill)."""
    if condition == "continue":
        question = data["explicit_q"]
        prefill = ""
    else:
        question = data["conditions"][condition]["question"]
        prefill = data["conditions"][condition]["prefill"]
    head = f"{question}\n\n" if question else ""
    return f"Human: {head}{wrapped}\n\nAssistant:", prefill


@torch.no_grad()
def run_one(run: LensRun) -> tuple[dict, dict[str, pd.DataFrame]]:
    data_path, data, data_sha256 = common.load_experiment_data(EXPERIMENT_NAME)

    generic_ids = any_number_ids(run.tokenizer)

    rows = []
    for passage in data["passages"]:
        wrapped = textwrap.fill(passage["text"], passage["width"])
        first_line = wrapped.split("\n", 1)[0]
        count = len(first_line)
        expected_ids = common.number_target_ids(run.tokenizer, count)
        letter = common.number_word(count)[0]
        letter_ids = common.word_variant_ids(run.tokenizer, letter)

        for condition in CONDITIONS:
            prefix, prefill = build_prompt(condition, data, wrapped)
            full_text = prefix + prefill
            input_ids, _ = common.encode_with_offsets(run.tokenizer, full_text)
            final_position = len(input_ids) - 1
            all_positions = list(range(len(input_ids)))
            ids_tensor = torch.tensor(
                [input_ids], dtype=torch.long, device=run.lens_model.input_device
            )

            with common.ActivationRecorder(run.lens_model.layers, at=run.band) as recorder:
                clean_final = (
                    run.hf_model(input_ids=ids_tensor).logits[0, final_position].float().cpu()
                )
                residuals = {
                    layer: recorder.activations[layer][0].detach().float()
                    for layer in run.band
                }

            if condition == "letter":
                answer_rank, answer_prob = (
                    common.rank_and_prob(clean_final, letter_ids)
                    if letter_ids
                    else (None, None)
                )
            elif condition in ("none", "direct"):
                answer_rank, answer_prob = (
                    common.rank_and_prob(clean_final, expected_ids)
                    if expected_ids
                    else (None, None)
                )
            else:
                answer_rank, answer_prob = None, None

            for method in common.METHODS:
                best_expected = float("inf")
                best_generic = float("inf")
                for layer in run.band:
                    logits = run.lens_logits(residuals[layer], layer, method)
                    if expected_ids:
                        best_expected = min(
                            best_expected,
                            float(common.ranks_of(logits, expected_ids).min().item()),
                        )
                    best_generic = min(
                        best_generic,
                        float(common.ranks_of(logits, generic_ids).min().item()),
                    )
                    del logits
                rows.append({
                    "method": method,
                    "condition": condition,
                    "passage": passage["tag"],
                    "width": passage["width"],
                    "count": count,
                    "count_scorable": bool(expected_ids),
                    "n_positions": len(all_positions),
                    "expected_rank_bandmin": best_expected,
                    "anynumber_rank_bandmin": best_generic,
                    **{f"expected_hit{k}": best_expected <= k for k in K_VALUES},
                    **{f"anynumber_hit{k}": best_generic <= k for k in K_VALUES},
                    "answer_rank": answer_rank,
                    "answer_prob": answer_prob,
                    "greedy": run.tokenizer.decode([int(clean_final.argmax())]),
                })
            del residuals, ids_tensor, clean_final
        if run.device.type == "cuda":
            torch.cuda.empty_cache()

    scores = pd.DataFrame(rows)
    scorable = scores[scores["count_scorable"]]

    summary_rows = []
    for (method, condition), group in scorable.groupby(["method", "condition"], sort=False):
        entry = {
            "method": method,
            "condition": condition,
            "n_passages": len(group),
            "expected_rank_median": float(group["expected_rank_bandmin"].median()),
            "anynumber_rank_median": float(group["anynumber_rank_bandmin"].median()),
        }
        for k in K_VALUES:
            rate, stderr = common.mean_stderr(group[f"expected_hit{k}"].astype(float))
            entry[f"expected_hit{k}_rate"] = rate
            entry[f"expected_hit{k}_stderr"] = stderr
            entry[f"anynumber_hit{k}_rate"] = float(group[f"anynumber_hit{k}"].mean())
        answers = group["answer_rank"].dropna()
        entry["answer_rank_median"] = float(answers.median()) if len(answers) else float("nan")
        entry["answer_prob_mean"] = (
            float(group["answer_prob"].dropna().mean())
            if group["answer_prob"].notna().any()
            else float("nan")
        )
        summary_rows.append(entry)

    def hit_rate(method: str, condition: str, k: int) -> float:
        match = [
            row[f"expected_hit{k}_rate"]
            for row in summary_rows
            if row["method"] == method and row["condition"] == condition
        ]
        return match[0] if match else float("nan")

    payload = {
        "dataset_sha256": data_sha256,
        "k_values": K_VALUES,
        "conditions": CONDITIONS,
        "n_generic_number_tokens": len(generic_ids),
        "count_coverage": float(scores.drop_duplicates("passage")["count_scorable"].mean()),
        "run_summary": summary_rows,
        "passage_scores": scores.to_dict(orient="records"),
        "headline": (
            f"expected@10 jac none={hit_rate('jacobian', 'none', 10):.3f} "
            f"direct={hit_rate('jacobian', 'direct', 10):.3f} "
            f"letter={hit_rate('jacobian', 'letter', 10):.3f}"
        ),
    }
    return payload, {"_trial_scores.csv.gz": scores}


if __name__ == "__main__":
    raise SystemExit(common.run_experiment(EXPERIMENT_NAME, run_one))
