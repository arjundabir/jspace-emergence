from __future__ import annotations

import pandas as pd
import torch

from experiments import common
from experiments.common import LensRun

EXPERIMENT_NAME = "top-down-summoning"
K_VALUES = [1, 5, 10]
ALPHAS = [1.0, 2.0]


def build_prompt(stimulus: str, question: str) -> tuple[str, int, int]:
    """Prompt text plus the char span of the stimulus inside it."""
    head = f"Human: {question}\n\nHere is the passage:\n\n"
    prompt = f"{head}{stimulus}\n\nAssistant:"
    return prompt, len(head), len(head) + len(stimulus)


def word_set_ids(tokenizer, words: list[str]) -> list[int]:
    ids: list[int] = []
    for word in words:
        ids += common.word_variant_ids(tokenizer, word)
    return sorted(set(ids))


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

    lens_rows = []
    swap_rows = []
    n_swaps_skipped = 0

    for item in data["items"]:
        questions = {"q1": data["q1"], "q2": item["q2"]}
        expected_ids = word_set_ids(run.tokenizer, item["expected"])
        foil_ids = word_set_ids(run.tokenizer, item["foil"])
        q1_expect_ids = word_set_ids(run.tokenizer, item["q1_expect"])
        if not expected_ids or not foil_ids:
            continue  # unscorable item: excluded, not a failure

        for question_key, question in questions.items():
            prompt, stim_start, stim_end = build_prompt(item["stimulus"], question)
            input_ids, offsets = common.encode_with_offsets(run.tokenizer, prompt)
            stimulus_positions = [
                index
                for index, (start, end) in enumerate(offsets)
                if start < stim_end and end > stim_start and end > start
            ]
            final_position = len(input_ids) - 1
            ids_tensor = torch.tensor(
                [input_ids], dtype=torch.long, device=run.lens_model.input_device
            )

            # Clean pass: model logits plus band residuals in one forward.
            with common.ActivationRecorder(run.lens_model.layers, at=run.band) as recorder:
                clean_final = (
                    run.hf_model(input_ids=ids_tensor).logits[0, final_position].float().cpu()
                )
                residuals = {
                    layer: recorder.activations[layer][0, stimulus_positions].detach().float()
                    for layer in run.band
                }

            expected_rank, expected_prob = common.rank_and_prob(clean_final, expected_ids)
            foil_rank, foil_prob = common.rank_and_prob(clean_final, foil_ids)
            q1_rank, q1_prob = (
                common.rank_and_prob(clean_final, q1_expect_ids)
                if q1_expect_ids
                else (None, None)
            )

            for method in common.METHODS:
                n_positions = len(stimulus_positions)
                band_min_expected = torch.full((n_positions,), float("inf"))
                band_min_foil = torch.full((n_positions,), float("inf"))
                for layer in run.band:
                    logits = run.lens_logits(residuals[layer], layer, method)
                    band_min_expected = torch.minimum(
                        band_min_expected,
                        common.ranks_of(logits, expected_ids).min(dim=1).values.float(),
                    )
                    band_min_foil = torch.minimum(
                        band_min_foil,
                        common.ranks_of(logits, foil_ids).min(dim=1).values.float(),
                    )
                    del logits
                entry = {
                    "method": method,
                    "item": item["key"],
                    "question": question_key,
                    "n_stimulus_positions": n_positions,
                    "expected_rank_bandmin": float(band_min_expected.min().item()),
                    "foil_rank_bandmin": float(band_min_foil.min().item()),
                    "answer_expected_rank": expected_rank,
                    "answer_expected_prob": expected_prob,
                    "answer_foil_rank": foil_rank,
                    "answer_foil_prob": foil_prob,
                    "answer_q1_expect_rank": q1_rank,
                    "answer_q1_expect_prob": q1_prob,
                    "greedy": run.tokenizer.decode([int(clean_final.argmax())]),
                }
                for k in K_VALUES:
                    entry[f"expected_posfrac{k}"] = float(
                        (band_min_expected <= k).float().mean().item()
                    )
                    entry[f"foil_posfrac{k}"] = float(
                        (band_min_foil <= k).float().mean().item()
                    )
                lens_rows.append(entry)
            del residuals

            # Causal swaps: label -> foil-label at every stimulus position.
            for label, foil_label in item["swaps"]:
                source_token, _ = common.preferred_single_token(run.tokenizer, label)
                target_token, _ = common.preferred_single_token(run.tokenizer, foil_label)
                if source_token is None or target_token is None:
                    n_swaps_skipped += 1
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
                            layer: common.make_swap_hook(
                                v_source, v_target, stimulus_positions, alpha
                            )
                            for layer, (v_source, v_target) in vectors.items()
                        }
                        swapped = common.forward_with_hooks(
                            run.hf_model, run.lens_model, ids_tensor, hooks
                        )[final_position].cpu()
                        expected_rank_after, expected_prob_after = common.rank_and_prob(
                            swapped, expected_ids
                        )
                        foil_rank_after, foil_prob_after = common.rank_and_prob(
                            swapped, foil_ids
                        )
                        top1_after = int(swapped.argmax())
                        swap_rows.append({
                            "method": method,
                            "alpha": alpha,
                            "item": item["key"],
                            "question": question_key,
                            "swap": f"{label}->{foil_label}",
                            "expected_rank_clean": expected_rank,
                            "expected_rank_after": expected_rank_after,
                            "expected_prob_clean": expected_prob,
                            "expected_prob_after": expected_prob_after,
                            "foil_rank_clean": foil_rank,
                            "foil_rank_after": foil_rank_after,
                            "foil_prob_clean": foil_prob,
                            "foil_prob_after": foil_prob_after,
                            "moved_to_foil": top1_after in foil_ids,
                            "top1_after": run.tokenizer.decode([top1_after]),
                        })
            del ids_tensor, clean_final
        if run.device.type == "cuda":
            torch.cuda.empty_cache()

    lens_scores = pd.DataFrame(lens_rows)
    swaps = pd.DataFrame(swap_rows)

    # Summoning effect: Q2 - Q1 position fraction, paired per item.
    summary_rows = []
    pivot = lens_scores.pivot_table(
        index=["method", "item"], columns="question",
        values=[f"expected_posfrac{k}" for k in K_VALUES],
        aggfunc="first",
    )
    for method in common.METHODS:
        method_rows = pivot.loc[method] if method in pivot.index else pd.DataFrame()
        entry = {"method": method, "n_items": len(method_rows)}
        for k in K_VALUES:
            column = f"expected_posfrac{k}"
            if method_rows.empty or (column, "q2") not in method_rows.columns:
                entry[f"summoning_effect_k{k}"] = float("nan")
                continue
            effect = method_rows[(column, "q2")] - method_rows[(column, "q1")]
            mean, stderr = common.mean_stderr(effect)
            entry[f"summoning_effect_k{k}"] = mean
            entry[f"summoning_effect_k{k}_stderr"] = stderr
            entry[f"q2_posfrac{k}_mean"] = float(method_rows[(column, "q2")].mean())
            entry[f"q1_posfrac{k}_mean"] = float(method_rows[(column, "q1")].mean())
        if not swaps.empty:
            for alpha in ALPHAS:
                q2_swaps = swaps[
                    (swaps["method"] == method)
                    & (swaps["alpha"] == alpha)
                    & (swaps["question"] == "q2")
                ]
                entry[f"swap_moved_to_foil_rate_q2_a{alpha:g}"] = (
                    float(q2_swaps["moved_to_foil"].mean()) if len(q2_swaps) else float("nan")
                )
        summary_rows.append(entry)

    def effect(method: str, k: int) -> float:
        match = [
            row.get(f"summoning_effect_k{k}")
            for row in summary_rows
            if row["method"] == method
        ]
        return match[0] if match else float("nan")

    payload = {
        "dataset_sha256": data_sha256,
        "alphas": ALPHAS,
        "k_values": K_VALUES,
        "n_items": int(lens_scores["item"].nunique()),
        "n_swaps_skipped": n_swaps_skipped,
        "run_summary": summary_rows,
        "item_scores": lens_scores.to_dict(orient="records"),
        "headline": (
            f"summoning k=10 jac={effect('jacobian', 10):.3f} "
            f"logit={effect('logit', 10):.3f}"
        ),
    }
    return payload, {"_swap_scores.csv.gz": swaps}


if __name__ == "__main__":
    raise SystemExit(common.run_experiment(EXPERIMENT_NAME, run_one))
