"""Verbal-report experiment (paper §3.1, Figures 6 & 7).

For each of 14 categories the model is prompted ``Think of a {category}.
Answer in one word.``; the readout position is the final ``:``. The greedy
next token is the model's answer and becomes the swap-out target: for each of
the first 10 listed candidates (skipping the answer itself), the answer's
representation is swapped for the candidate's — the lens-coordinate swap
clamped at every prompt position across the workspace band — and the
swapped-in candidate's rank and probability in the output distribution are
recorded. Success is the candidate at rank 1.

The swap-out direction is the greedy token's own lens vector, whatever the
model produced — on early checkpoints that token is typically punctuation,
which is the expected floor (``answer_is_candidate`` marks whether the clean
answer was itself a category word). Candidates with no single-token form are
excluded from the denominator.

    python -m experiments.verbal_report
"""

from __future__ import annotations

import pandas as pd
import torch

from experiments import common
from experiments.common import LensRun

EXPERIMENT_NAME = "verbal-report"
K_VALUES = [1, 5, 10]
N_CANDIDATES = 10
ALPHAS = [1.0, 2.0]


@torch.no_grad()
def run_one(run: LensRun) -> tuple[dict, dict[str, pd.DataFrame]]:
    data_path, data, data_sha256 = common.load_experiment_data(EXPERIMENT_NAME)

    vector_cache: dict[tuple, torch.Tensor] = {}

    def cached_lens_vector(layer: int, token_id: int, method: str):
        key = (layer, token_id, method)
        if key not in vector_cache:
            vector_cache[key] = common.lens_vector(
                run.lens, run.lens_model, layer, token_id, method, normalize=False
            )
        return vector_cache[key]

    category_rows = []
    swap_rows = []
    n_candidates_skipped = 0

    for category, candidates in data["candidates"].items():
        prompt = f"Human: Think of a {category}. Answer in one word.\n\nAssistant:"
        input_ids, _ = common.encode_with_offsets(run.tokenizer, prompt)
        final_position = len(input_ids) - 1
        all_positions = list(range(len(input_ids)))
        ids_tensor = torch.tensor(
            [input_ids], dtype=torch.long, device=run.lens_model.input_device
        )

        clean_final = (
            run.hf_model(input_ids=ids_tensor).logits[0, final_position].float().cpu()
        )
        answer_token = int(clean_final.argmax())
        answer_surface = run.tokenizer.decode([answer_token])
        _, answer_prob = common.rank_and_prob(clean_final, [answer_token])
        answer_word = answer_surface.strip().lower()
        answer_is_candidate = answer_word in {c.lower() for c in candidates}

        category_rows.append({
            "category": category,
            "answer_surface": answer_surface,
            "answer_prob": answer_prob,
            "answer_is_candidate": answer_is_candidate,
        })

        swap_ins = [c for c in candidates if c.lower() != answer_word]
        swap_ins = swap_ins[:N_CANDIDATES]

        for candidate in swap_ins:
            target_token, _ = common.preferred_single_token(run.tokenizer, candidate)
            if target_token is None:
                n_candidates_skipped += 1
                continue
            candidate_ids = common.word_variant_ids(run.tokenizer, candidate)
            rank_clean, prob_clean = common.rank_and_prob(clean_final, candidate_ids)

            for method in common.METHODS:
                vectors = {
                    layer: (
                        cached_lens_vector(layer, answer_token, method),
                        cached_lens_vector(layer, target_token, method),
                    )
                    for layer in run.band
                }
                for alpha in ALPHAS:
                    hooks = {
                        layer: common.make_swap_hook(v_source, v_target, all_positions, alpha)
                        for layer, (v_source, v_target) in vectors.items()
                    }
                    swapped = common.forward_with_hooks(
                        run.hf_model, run.lens_model, ids_tensor, hooks
                    )[final_position].cpu()
                    rank_after, prob_after = common.rank_and_prob(swapped, candidate_ids)
                    answer_rank_after, answer_prob_after = common.rank_and_prob(
                        swapped, [answer_token]
                    )
                    top1_after = int(swapped.argmax())
                    swap_rows.append({
                        "method": method,
                        "alpha": alpha,
                        "category": category,
                        "answer_surface": answer_surface,
                        "answer_is_candidate": answer_is_candidate,
                        "candidate": candidate,
                        "candidate_rank_clean": rank_clean,
                        "candidate_rank_after": rank_after,
                        "candidate_prob_clean": prob_clean,
                        "candidate_prob_after": prob_after,
                        "answer_rank_after": answer_rank_after,
                        "answer_prob_clean": answer_prob,
                        "answer_prob_after": answer_prob_after,
                        "success": top1_after in candidate_ids,
                        "answer_retained": top1_after == answer_token,
                        "top1_after": run.tokenizer.decode([top1_after]),
                        **{f"hit{k}": rank_after <= k for k in K_VALUES},
                    })
        del ids_tensor, clean_final
        if run.device.type == "cuda":
            torch.cuda.empty_cache()

    categories = pd.DataFrame(category_rows)
    swaps = pd.DataFrame(swap_rows)

    answer_is_candidate_rate = float(categories["answer_is_candidate"].mean())

    summary_rows = []
    for (method, alpha), group in swaps.groupby(["method", "alpha"], sort=False):
        success_rate, success_stderr = common.mean_stderr(group["success"].astype(float))
        entry = {
            "method": method,
            "alpha": float(alpha),
            "n_swaps": len(group),
            "swap_success_rate": success_rate,
            "swap_success_stderr": success_stderr,
            "candidate_rank_clean_median": float(group["candidate_rank_clean"].median()),
            "candidate_rank_after_median": float(group["candidate_rank_after"].median()),
            "candidate_prob_gain_mean": float(
                (group["candidate_prob_after"] - group["candidate_prob_clean"]).mean()
            ),
            "answer_retained_rate": float(group["answer_retained"].mean()),
            "answer_is_candidate_rate": answer_is_candidate_rate,
        }
        for k in K_VALUES:
            entry[f"hit{k}_rate"] = float(group[f"hit{k}"].mean())
        summary_rows.append(entry)

    category_summary = [
        {
            "method": method,
            "alpha": float(alpha),
            "category": category,
            "n_swaps": len(group),
            "swap_success_rate": float(group["success"].mean()),
            "candidate_rank_after_median": float(group["candidate_rank_after"].median()),
        }
        for (method, alpha, category), group in swaps.groupby(
            ["method", "alpha", "category"], sort=False
        )
    ]

    def rate(method: str, alpha: float) -> float:
        match = [
            row["swap_success_rate"]
            for row in summary_rows
            if row["method"] == method and row["alpha"] == alpha
        ]
        return match[0] if match else float("nan")

    payload = {
        "dataset_sha256": data_sha256,
        "alphas": ALPHAS,
        "k_values": K_VALUES,
        "n_candidates_per_category": N_CANDIDATES,
        "n_categories": len(categories),
        "n_candidates_skipped": n_candidates_skipped,
        "answer_is_candidate_rate": answer_is_candidate_rate,
        "run_summary": summary_rows,
        "category_summary": category_summary,
        "category_answers": categories.to_dict(orient="records"),
        "headline": (
            f"answer-is-candidate={answer_is_candidate_rate:.2f} "
            f"swap@a=1 jac={rate('jacobian', 1.0):.3f} "
            f"logit={rate('logit', 1.0):.3f}"
        ),
    }
    return payload, {"_swap_scores.csv.gz": swaps}


if __name__ == "__main__":
    raise SystemExit(common.run_experiment(EXPERIMENT_NAME, run_one))
