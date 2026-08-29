from __future__ import annotations

import pandas as pd
import torch

from experiments import common
from experiments.common import ActivationRecorder, LensRun

EXPERIMENT_NAME = "flexible-generalization"
ALPHAS = [1.0, 2.0]


def answer_token_ids(tokenizer, answer: str) -> tuple[list[int], bool]:
    """Single-token ids of ``answer`` (bare + leading-space forms).

    Falls back to the first token of the leading-space tokenization when no
    form is a single token, flagged so those items can be filtered.
    """
    forms = common.single_token_forms(tokenizer, answer)
    if forms:
        return sorted(set(forms.values())), True
    ids = tokenizer.encode(" " + answer, add_special_tokens=False)
    return [ids[0]], False


def arg_token_span(offsets: list[tuple[int, int]], char_start: int, char_end: int) -> list[int]:
    return [
        index
        for index, (start, end) in enumerate(offsets)
        if start < char_end and end > char_start and end > start
    ]


@torch.no_grad()
def run_one(run: LensRun) -> tuple[dict, dict[str, pd.DataFrame]]:
    data_path, data, data_sha256 = common.load_experiment_data(EXPERIMENT_NAME)

    # Each (layer, token) lens vector costs a d x d matvec and is reused
    # across targets, alphas, and the loading measurement; cache per lens.
    vector_cache: dict[tuple, torch.Tensor] = {}

    def cached_lens_vector(layer: int, token_id: int, method: str, normalize: bool):
        key = (layer, token_id, method, normalize)
        if key not in vector_cache:
            vector_cache[key] = common.lens_vector(
                run.lens, run.lens_model, layer, token_id, method, normalize
            )
        return vector_cache[key]

    baseline_rows = []
    swap_rows = []

    for category in data["categories"]:
        for func in category["funcs"]:
            template = func["template"]
            prefix_text = template.split("{arg}")[0]
            for source_arg in category["args"]:
                prompt = template.format(arg=source_arg)
                input_ids, offsets = common.encode_with_offsets(run.tokenizer, prompt)
                arg_span = arg_token_span(
                    offsets, len(prefix_text), len(prefix_text) + len(source_arg)
                )
                final_position = len(input_ids) - 1
                ids_tensor = torch.tensor(
                    [input_ids], dtype=torch.long, device=run.lens_model.input_device
                )

                # Clean pass: model logits plus band residuals in one forward.
                with ActivationRecorder(run.lens_model.layers, at=run.band) as recorder:
                    clean_logits = run.hf_model(input_ids=ids_tensor).logits[0].float().cpu()
                    loading_positions = sorted(set(arg_span + [final_position]))
                    band_residuals = {
                        layer: recorder.activations[layer][0, loading_positions]
                        .detach().float().cpu()
                        for layer in run.band
                    }
                clean_final = clean_logits[final_position]
                model_top1 = int(clean_final.argmax())

                answer = func["answers"][source_arg]
                answer_ids, answer_scorable = answer_token_ids(run.tokenizer, answer)
                answer_rank = min(
                    common.one_based_rank(clean_final, token_id) for token_id in answer_ids
                )

                # The in-context argument token: single-token args only for
                # the swap (the lens names single-token concepts).
                source_token = input_ids[arg_span[0]] if arg_span else None
                source_swappable = len(arg_span) == 1
                source_surface = (
                    run.tokenizer.decode([source_token]) if source_swappable else ""
                )

                loading = {}
                for method in common.METHODS:
                    if not source_swappable:
                        loading[method] = float("nan")
                        continue
                    cosines = []
                    for layer in run.band:
                        vector = cached_lens_vector(layer, source_token, method, normalize=True)
                        residual = band_residuals[layer]
                        cosines.append(
                            torch.nn.functional.cosine_similarity(
                                residual, vector.unsqueeze(0), dim=-1
                            ).mean().item()
                        )
                    loading[method] = float(sum(cosines) / len(cosines))

                baseline_rows.append({
                    "category": category["name"],
                    "func": func["name"],
                    "arg": source_arg,
                    "answer": answer,
                    "answer_scorable": answer_scorable,
                    "answer_rank": answer_rank,
                    "answer_top1": model_top1 in answer_ids,
                    "model_top1": run.tokenizer.decode([model_top1]),
                    "arg_span_n_tokens": len(arg_span),
                    "source_surface": source_surface,
                    "loading_jacobian": loading.get("jacobian", float("nan")),
                    "loading_logit": loading.get("logit", float("nan")),
                })

                if not source_swappable:
                    continue

                leading_space = source_surface.startswith(" ")
                swap_positions = list(range(len(input_ids)))
                for target_arg in category["args"]:
                    if target_arg == source_arg:
                        continue
                    target_form = (" " if leading_space else "") + target_arg
                    target_ids = run.tokenizer.encode(target_form, add_special_tokens=False)
                    if len(target_ids) != 1:
                        continue  # no single-token lens vector to swap in
                    target_token = target_ids[0]
                    target_answer = func["answers"][target_arg]
                    target_answer_ids, target_scorable = answer_token_ids(
                        run.tokenizer, target_answer
                    )
                    rank_clean = min(
                        common.one_based_rank(clean_final, token_id)
                        for token_id in target_answer_ids
                    )

                    for method in common.METHODS:
                        vectors = {
                            layer: (
                                cached_lens_vector(layer, source_token, method, normalize=False),
                                cached_lens_vector(layer, target_token, method, normalize=False),
                            )
                            for layer in run.band
                        }
                        for alpha in ALPHAS:
                            hooks = {
                                layer: common.make_swap_hook(
                                    v_source, v_target, swap_positions, alpha
                                )
                                for layer, (v_source, v_target) in vectors.items()
                            }
                            swapped = common.forward_with_hooks(
                                run.hf_model, run.lens_model, ids_tensor, hooks
                            )[final_position].cpu()
                            rank_after = min(
                                common.one_based_rank(swapped, token_id)
                                for token_id in target_answer_ids
                            )
                            top1_after = int(swapped.argmax())
                            swap_rows.append({
                                "method": method,
                                "alpha": alpha,
                                "category": category["name"],
                                "func": func["name"],
                                "source_arg": source_arg,
                                "target_arg": target_arg,
                                "target_answer": target_answer,
                                "target_answer_scorable": target_scorable,
                                "target_answer_rank_clean": rank_clean,
                                "target_answer_rank_after": rank_after,
                                "success": top1_after in target_answer_ids,
                                "top1_after": run.tokenizer.decode([top1_after]),
                                "loading_source": loading[method],
                            })
                del ids_tensor, clean_logits, band_residuals
            if run.device.type == "cuda":
                torch.cuda.empty_cache()

    baseline = pd.DataFrame(baseline_rows)
    swaps = pd.DataFrame(swap_rows)

    # Paper-metric denominators contain only answers with a single-token
    # surface; the first-token-proxy rows stay in the CSV, flagged.
    scorable_baseline = baseline[baseline["answer_scorable"]]
    scorable_swaps = swaps[swaps["target_answer_scorable"]]
    baseline_top1_rate = float(scorable_baseline["answer_top1"].mean())
    answer_coverage = float(baseline["answer_scorable"].mean())

    summary_rows = []
    for (method, alpha), group in scorable_swaps.groupby(["method", "alpha"], sort=False):
        success_rate, success_stderr = common.mean_stderr(group["success"].astype(float))
        summary_rows.append({
            "method": method,
            "alpha": float(alpha),
            "n_swaps_scorable": len(group),
            "n_swaps_total": int(
                (swaps["method"] == method).mul(swaps["alpha"] == alpha).sum()
            ),
            "swap_success_rate": success_rate,
            "swap_success_stderr": success_stderr,
            "target_answer_rank_after_median": float(group["target_answer_rank_after"].median()),
            "target_answer_rank_clean_median": float(group["target_answer_rank_clean"].median()),
            "baseline_answer_top1_rate": baseline_top1_rate,
            "answer_tokenization_coverage": answer_coverage,
        })

    function_summary = [
        {
            "method": method,
            "alpha": float(alpha),
            "category": category,
            "func": func,
            "n_swaps": len(group),
            "swap_success_rate": float(group["success"].mean()),
            "loading_source_mean": float(group["loading_source"].mean()),
        }
        for (method, alpha, category, func), group in scorable_swaps.groupby(
            ["method", "alpha", "category", "func"], sort=False
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
        "n_baseline_items": len(baseline),
        "n_baseline_items_scorable": len(scorable_baseline),
        "answer_tokenization_coverage": answer_coverage,
        "baseline_answer_top1_rate": baseline_top1_rate,
        "run_summary": summary_rows,
        "function_summary": function_summary,
        "baseline_items": baseline.to_dict(orient="records"),
        "headline": (
            f"baseline top1={baseline_top1_rate:.3f} "
            f"swap@a=1 jac={rate('jacobian', 1.0):.3f} logit={rate('logit', 1.0):.3f} "
            f"(coverage={answer_coverage:.2f})"
        ),
    }
    return payload, {"_swap_scores.csv.gz": swaps}


if __name__ == "__main__":
    raise SystemExit(common.run_experiment(EXPERIMENT_NAME, run_one))
