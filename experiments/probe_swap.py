from __future__ import annotations

import pandas as pd
import torch

from experiments import common
from experiments.common import LensRun

EXPERIMENT_NAME = "probe-swap"
K_VALUES = [1, 5, 10]
ALPHAS = [1.0, 2.0]


def completion_boundary(tokenizer, prompt: str, completion: str) -> tuple[list[int], int]:
    separator = "" if prompt.endswith((" ", "\n")) or completion.startswith(" ") else " "
    input_ids, offsets = common.encode_with_offsets(
        tokenizer, prompt + separator + completion
    )
    boundary = len(prompt)
    index = next(
        i for i, (start, end) in enumerate(offsets) if end > boundary and end > start
    )
    return input_ids[:index], input_ids[index]


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

    item_rows = []
    lens_rows = []
    swap_rows = []
    n_prefix_mismatch = 0

    for item in data["items"]:
        prompt = item["prompt"]
        prefix_ids, answer_token = completion_boundary(
            run.tokenizer, prompt, item["answer"]
        )
        swap_prefix_ids, swap_answer_token = completion_boundary(
            run.tokenizer, prompt, item["swap_answer"]
        )
        # Both completions must sit on the identical prompt-side prefix for
        # the clean/after ranks to be comparable at one readout position.
        comparable = swap_prefix_ids == prefix_ids
        if not comparable:
            n_prefix_mismatch += 1

        final_position = len(prefix_ids) - 1
        all_positions = list(range(len(prefix_ids)))
        ids_tensor = torch.tensor(
            [prefix_ids], dtype=torch.long, device=run.lens_model.input_device
        )

        # Clean pass: model logits plus band residuals in one forward.
        with common.ActivationRecorder(run.lens_model.layers, at=run.band) as recorder:
            clean_logits = run.hf_model(input_ids=ids_tensor).logits[0].float().cpu()
            residuals = {
                layer: recorder.activations[layer][0].detach().float()
                for layer in run.band
            }
        clean_final = clean_logits[final_position]
        model_top1 = int(clean_final.argmax())

        answer_rank, answer_prob = common.rank_and_prob(clean_final, [answer_token])
        swap_answer_rank, swap_answer_prob = common.rank_and_prob(
            clean_final, [swap_answer_token]
        )

        # Lens readout of the latent intermediate over the prompt span.
        intermediate_ids = common.word_variant_ids(run.tokenizer, item["intermediate"])
        band_min: dict[str, float] = {}
        for method in common.METHODS:
            best = float("inf")
            for layer in run.band:
                if not intermediate_ids:
                    break
                logits = run.lens_logits(residuals[layer], layer, method)
                layer_min = common.ranks_of(logits, intermediate_ids).min().item()
                lens_rows.append({
                    "method": method,
                    "name": item["name"],
                    "category": item["category"],
                    "layer": layer,
                    "intermediate_rank_min": int(layer_min),
                })
                best = min(best, float(layer_min))
                del logits
            band_min[method] = best
        del residuals

        item_rows.append({
            "name": item["name"],
            "category": item["category"],
            "intermediate": item["intermediate"],
            "swap_to": item["swap_to"],
            "answer": item["answer"],
            "swap_answer": item["swap_answer"],
            "answer_rank_clean": answer_rank,
            "answer_prob_clean": answer_prob,
            "swap_answer_rank_clean": swap_answer_rank,
            "swap_answer_prob_clean": swap_answer_prob,
            "baseline_correct": model_top1 == answer_token,
            "model_top1": run.tokenizer.decode([model_top1]),
            "intermediate_scorable": bool(intermediate_ids),
            "intermediate_rank_bandmin_jacobian": band_min.get("jacobian", float("inf")),
            "intermediate_rank_bandmin_logit": band_min.get("logit", float("inf")),
            "prefix_comparable": comparable,
        })

        # Swap needs single-token lens vectors for both the intermediate and
        # its replacement, and a comparable readout position.
        source_token, _ = common.preferred_single_token(run.tokenizer, item["intermediate"])
        target_token, _ = common.preferred_single_token(run.tokenizer, item["swap_to"])
        if source_token is None or target_token is None or not comparable:
            del ids_tensor, clean_logits
            continue

        for method in common.METHODS:
            vectors = {
                layer: (
                    cached_lens_vector(layer, source_token, method),
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
                answer_rank_after, answer_prob_after = common.rank_and_prob(
                    swapped, [answer_token]
                )
                swap_rank_after, swap_prob_after = common.rank_and_prob(
                    swapped, [swap_answer_token]
                )
                top1_after = int(swapped.argmax())
                swap_rows.append({
                    "method": method,
                    "alpha": alpha,
                    "name": item["name"],
                    "category": item["category"],
                    "answer_rank_clean": answer_rank,
                    "answer_rank_after": answer_rank_after,
                    "answer_prob_clean": answer_prob,
                    "answer_prob_after": answer_prob_after,
                    "swap_answer_rank_clean": swap_answer_rank,
                    "swap_answer_rank_after": swap_rank_after,
                    "swap_answer_prob_clean": swap_answer_prob,
                    "swap_answer_prob_after": swap_prob_after,
                    "success": top1_after == swap_answer_token,
                    "baseline_correct": model_top1 == answer_token,
                    "top1_after": run.tokenizer.decode([top1_after]),
                })
        del ids_tensor, clean_logits
        if run.device.type == "cuda":
            torch.cuda.empty_cache()

    items = pd.DataFrame(item_rows)
    swaps = pd.DataFrame(swap_rows)
    lens_scores = pd.DataFrame(lens_rows)

    baseline_correct_rate = float(items["baseline_correct"].mean())
    scorable = items[items["intermediate_scorable"]]

    summary_rows = []
    for method in common.METHODS:
        column = f"intermediate_rank_bandmin_{method}"
        recovery = {
            f"intermediate_hit{k}_rate": float((scorable[column] <= k).mean())
            for k in K_VALUES
        }
        for alpha in ALPHAS:
            group = swaps[(swaps["method"] == method) & (swaps["alpha"] == alpha)]
            success_rate, success_stderr = common.mean_stderr(group["success"].astype(float))
            summary_rows.append({
                "method": method,
                "alpha": float(alpha),
                "n_items": len(items),
                "n_swaps": len(group),
                "baseline_correct_rate": baseline_correct_rate,
                **recovery,
                "swap_success_rate": success_rate,
                "swap_success_stderr": success_stderr,
                "swap_answer_rank_clean_median": float(group["swap_answer_rank_clean"].median()),
                "swap_answer_rank_after_median": float(group["swap_answer_rank_after"].median()),
                "answer_rank_after_median": float(group["answer_rank_after"].median()),
                "swap_answer_prob_gain_mean": float(
                    (group["swap_answer_prob_after"] - group["swap_answer_prob_clean"]).mean()
                ),
            })

    category_summary = [
        {
            "method": method,
            "alpha": float(alpha),
            "category": category,
            "n_swaps": len(group),
            "swap_success_rate": float(group["success"].mean()),
            "baseline_correct_rate": float(group["baseline_correct"].mean()),
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
        "n_items": len(items),
        "n_prefix_mismatch": n_prefix_mismatch,
        "n_swappable_items": int(swaps["name"].nunique()),
        "swap_directions": "lens vectors (paper's linear probes are not released)",
        "run_summary": summary_rows,
        "category_summary": category_summary,
        "baseline_items": items.to_dict(orient="records"),
        "headline": (
            f"baseline={baseline_correct_rate:.3f} "
            f"swap@a=1 jac={rate('jacobian', 1.0):.3f} "
            f"logit={rate('logit', 1.0):.3f}"
        ),
    }
    return payload, {
        "_swap_scores.csv.gz": swaps,
        "_lens_layer_ranks.csv.gz": lens_scores,
    }


if __name__ == "__main__":
    raise SystemExit(common.run_experiment(EXPERIMENT_NAME, run_one))
