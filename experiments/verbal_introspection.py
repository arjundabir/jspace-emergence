from __future__ import annotations

import pandas as pd
import torch

from experiments import common
from experiments.common import LensRun

EXPERIMENT_NAME = "verbal-introspection"
K_VALUES = [1, 5, 10]
ROLE_LABELS = {"user": "Human:", "assistant": "Assistant:"}
STRENGTHS = [0.0, 1.0, 2.0, 4.0, 8.0]
PREFILL_KEYS = ["default"]


def conversation_text(turns: list[dict]) -> tuple[str, tuple[int, int]]:
    parts: list[str] = []
    question_span = (0, 0)
    cursor = 0
    for index, turn in enumerate(turns):
        label = ROLE_LABELS[turn["role"]]
        if index:
            cursor += len("\n\n")
        content_start = cursor + len(label)
        content_end = content_start + len(turn["content"])
        if turn["role"] == "user":
            question_span = (content_start, content_end)
        parts.append(label + turn["content"])
        cursor = content_end
    return "\n\n".join(parts), question_span


def make_inject_hook(vector: torch.Tensor, positions: list[int], scale: float):
    def hook(module, inputs, output):
        hidden = output if torch.is_tensor(output) else output[0]
        hidden = hidden.clone()
        delta = scale * vector.to(hidden.device, torch.float32)
        block = hidden[:, positions, :].float() + delta
        hidden[:, positions, :] = block.to(hidden.dtype)
        if torch.is_tensor(output):
            return hidden
        return (hidden, *output[1:])

    return hook


@torch.no_grad()
def run_one(run: LensRun) -> tuple[dict, dict[str, pd.DataFrame]]:
    data_path, data, data_sha256 = common.load_experiment_data(EXPERIMENT_NAME)
    concepts = data["concepts"]
    prompt_text, question_span = conversation_text(data["intro_prompt"])

    rows = []
    n_concepts_skipped = 0
    for prefill_key in PREFILL_KEYS:
        full_text = prompt_text + data["prefills"][prefill_key]
        input_ids, offsets = common.encode_with_offsets(run.tokenizer, full_text)
        final_position = len(input_ids) - 1
        question_positions = [
            index
            for index, (start, end) in enumerate(offsets)
            if start < question_span[1] and end > question_span[0] and end > start
        ]
        ids_tensor = torch.tensor(
            [input_ids], dtype=torch.long, device=run.lens_model.input_device
        )

        # Clean pass: control logits plus the per-layer residual-norm scale,
        # both taken over the question-turn positions the injection targets.
        residuals = common.capture_band_residuals(run, input_ids, question_positions)
        mean_norms = {
            layer: float(residuals[layer].norm(dim=-1).mean().item())
            for layer in run.band
        }
        del residuals
        clean_final = (
            run.hf_model(input_ids=ids_tensor).logits[0, final_position].float().cpu()
        )

        for concept in concepts:
            surface = concept["surface"]
            steer_token, _ = common.preferred_single_token(run.tokenizer, surface)
            score_ids = sorted(
                set(common.single_token_forms(run.tokenizer, surface).values())
            )
            if steer_token is None or not score_ids:
                n_concepts_skipped += 1
                continue
            rank_control, prob_control = common.rank_and_prob(clean_final, score_ids)

            for method in common.METHODS:
                directions = {
                    layer: common.lens_vector(
                        run.lens, run.lens_model, layer, steer_token, method,
                        normalize=True,
                    )
                    for layer in run.band
                }
                for strength in STRENGTHS:
                    if strength == 0.0:
                        rank_after, prob_after = rank_control, prob_control
                    else:
                        hooks = {
                            layer: make_inject_hook(
                                direction,
                                question_positions,
                                strength * mean_norms[layer],
                            )
                            for layer, direction in directions.items()
                        }
                        injected = common.forward_with_hooks(
                            run.hf_model, run.lens_model, ids_tensor, hooks
                        )[final_position].cpu()
                        rank_after, prob_after = common.rank_and_prob(injected, score_ids)
                    rows.append({
                        "method": method,
                        "prefill": prefill_key,
                        "concept": concept["name"],
                        "strength": strength,
                        "rank": rank_after,
                        "reciprocal_rank": 1.0 / rank_after,
                        "prob": prob_after,
                        "rank_control": rank_control,
                        **{f"hit{k}": rank_after <= k for k in K_VALUES},
                    })
        del ids_tensor, clean_final
        if run.device.type == "cuda":
            torch.cuda.empty_cache()

    scores = pd.DataFrame(rows)

    summary_rows = []
    for (method, prefill, strength), group in scores.groupby(
        ["method", "prefill", "strength"], sort=False
    ):
        entry = {
            "method": method,
            "prefill": prefill,
            "strength": float(strength),
            "n_concepts": len(group),
            "median_reciprocal_rank": float(group["reciprocal_rank"].median()),
            "mean_reciprocal_rank": float(group["reciprocal_rank"].mean()),
            "median_rank": float(group["rank"].median()),
            "mean_prob": float(group["prob"].mean()),
        }
        for k in K_VALUES:
            rate, stderr = common.mean_stderr(group[f"hit{k}"].astype(float))
            entry[f"hit{k}_rate"] = rate
            entry[f"hit{k}_stderr"] = stderr
        summary_rows.append(entry)

    def mrr(method: str, strength: float) -> float:
        match = [
            row["median_reciprocal_rank"]
            for row in summary_rows
            if row["method"] == method and row["strength"] == strength
        ]
        return float(pd.Series(match).mean()) if match else float("nan")

    strongest = max(STRENGTHS)
    payload = {
        "dataset_sha256": data_sha256,
        "strengths": STRENGTHS,
        "prefills": PREFILL_KEYS,
        "k_values": K_VALUES,
        "n_concepts": len(concepts),
        "n_concepts_skipped": n_concepts_skipped,
        "n_question_positions_note": "injection spans every user-turn token",
        "run_summary": summary_rows,
        "headline": (
            f"median RR jac: s=0 {mrr('jacobian', 0.0):.4f} -> "
            f"s={strongest:g} {mrr('jacobian', strongest):.4f} "
            f"(logit {mrr('logit', strongest):.4f})"
        ),
    }
    return payload, {"_trial_scores.csv.gz": scores}


if __name__ == "__main__":
    raise SystemExit(common.run_experiment(EXPERIMENT_NAME, run_one))
