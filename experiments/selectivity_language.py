"""Language-selectivity experiment (paper §3.7, Figure 26).

Eight short passages (two per language: French, German, Spanish, Italian)
run under two conditions built from the same template pair —
``task.explicit_q`` (name an author who wrote in the passage's language: the
language must be *explicitly used*) and ``task.automatic_q`` (continue the
passage: the language is only processed *automatically*). The
``intermediates[category]`` label words are tracked in the lens over the
question tokens following the passage. Selectivity is the explicit −
automatic hit rate: workspace content should carry the language label mainly
when the task demands reporting it.

Behavioral check at the final prompt position: the minimum next-token rank
over the correct language's ``authors`` versus the best rank over the other
three languages' authors. Words with no single-token form are excluded from
their denominators.

    python -m experiments.selectivity_language
"""

from __future__ import annotations

import pandas as pd
import torch

from experiments import common
from experiments.common import LensRun

EXPERIMENT_NAME = "selectivity-language"
K_VALUES = [1, 5, 10]
CONDITIONS = ["explicit", "automatic"]


def word_set_ids(tokenizer, words_by_category: dict) -> dict[str, list[int]]:
    ids: dict[str, list[int]] = {}
    for category, words in words_by_category.items():
        category_ids: list[int] = []
        for word in words:
            category_ids += common.word_variant_ids(tokenizer, word)
        ids[category] = sorted(set(category_ids))
    return ids


@torch.no_grad()
def run_one(run: LensRun) -> tuple[dict, dict[str, pd.DataFrame]]:
    data_path, data, data_sha256 = common.load_experiment_data(EXPERIMENT_NAME)

    label_ids = word_set_ids(run.tokenizer, data["intermediates"])
    author_ids = word_set_ids(run.tokenizer, data["authors"])
    templates = {
        "explicit": data["task"]["explicit_q"],
        "automatic": data["task"]["automatic_q"],
    }

    rows = []
    for passage in data["passages"]:
        category = passage["category"]
        correct_labels = label_ids[category]
        other_labels = sorted({
            token_id
            for other, ids in label_ids.items()
            if other != category
            for token_id in ids
        })
        if not correct_labels:
            continue  # unscorable category: excluded, not a failure

        for condition in CONDITIONS:
            user_content = templates[condition].format(text=passage["text"])
            prompt = f"Human: {user_content}\n\nAssistant:"
            passage_start = prompt.index(passage["text"])
            question_start = passage_start + len(passage["text"])
            question_end = len("Human: ") + len(user_content)

            input_ids, offsets = common.encode_with_offsets(run.tokenizer, prompt)
            question_positions = [
                index
                for index, (start, end) in enumerate(offsets)
                if start < question_end and end > question_start and end > start
            ]
            final_position = len(input_ids) - 1
            ids_tensor = torch.tensor(
                [input_ids], dtype=torch.long, device=run.lens_model.input_device
            )

            with common.ActivationRecorder(run.lens_model.layers, at=run.band) as recorder:
                clean_final = (
                    run.hf_model(input_ids=ids_tensor).logits[0, final_position].float().cpu()
                )
                residuals = {
                    layer: recorder.activations[layer][0, question_positions].detach().float()
                    for layer in run.band
                }

            correct_author_rank, correct_author_prob = (
                common.rank_and_prob(clean_final, author_ids[category])
                if author_ids[category]
                else (None, None)
            )
            other_author_ranks = [
                common.rank_and_prob(clean_final, ids)[0]
                for other, ids in author_ids.items()
                if other != category and ids
            ]
            best_other_author_rank = min(other_author_ranks) if other_author_ranks else None
            author_favored = (
                correct_author_rank < best_other_author_rank
                if correct_author_rank is not None and best_other_author_rank is not None
                else None
            )

            for method in common.METHODS:
                best_correct = float("inf")
                best_other = float("inf")
                for layer in run.band:
                    logits = run.lens_logits(residuals[layer], layer, method)
                    best_correct = min(
                        best_correct,
                        float(common.ranks_of(logits, correct_labels).min().item()),
                    )
                    if other_labels:
                        best_other = min(
                            best_other,
                            float(common.ranks_of(logits, other_labels).min().item()),
                        )
                    del logits
                rows.append({
                    "method": method,
                    "condition": condition,
                    "category": category,
                    "passage": passage["key"],
                    "n_question_positions": len(question_positions),
                    "label_rank_bandmin": best_correct,
                    "other_label_rank_bandmin": best_other,
                    **{f"hit{k}": best_correct <= k for k in K_VALUES},
                    "correct_author_rank": correct_author_rank,
                    "correct_author_prob": correct_author_prob,
                    "best_other_author_rank": best_other_author_rank,
                    "author_favored": author_favored,
                    "greedy": run.tokenizer.decode([int(clean_final.argmax())]),
                })
            del residuals, ids_tensor, clean_final
        if run.device.type == "cuda":
            torch.cuda.empty_cache()

    scores = pd.DataFrame(rows)

    summary_rows = []
    for method in common.METHODS:
        method_scores = scores[scores["method"] == method]
        entry = {"method": method, "n_passages": int(method_scores["passage"].nunique())}
        for condition in CONDITIONS:
            group = method_scores[method_scores["condition"] == condition]
            for k in K_VALUES:
                entry[f"{condition}_hit{k}_rate"] = float(group[f"hit{k}"].mean())
            entry[f"{condition}_label_rank_median"] = float(group["label_rank_bandmin"].median())
            favored = group["author_favored"].dropna()
            entry[f"{condition}_author_favored_rate"] = (
                float(favored.astype(float).mean()) if len(favored) else float("nan")
            )
        for k in K_VALUES:
            entry[f"selectivity_k{k}"] = (
                entry[f"explicit_hit{k}_rate"] - entry[f"automatic_hit{k}_rate"]
            )
        summary_rows.append(entry)

    category_summary = [
        {
            "method": method,
            "condition": condition,
            "category": category,
            "n_passages": len(group),
            "hit1_rate": float(group["hit1"].mean()),
            "hit10_rate": float(group["hit10"].mean()),
            "label_rank_median": float(group["label_rank_bandmin"].median()),
        }
        for (method, condition, category), group in scores.groupby(
            ["method", "condition", "category"], sort=False
        )
    ]

    def selectivity(method: str, k: int) -> float:
        match = [row[f"selectivity_k{k}"] for row in summary_rows if row["method"] == method]
        return match[0] if match else float("nan")

    payload = {
        "dataset_sha256": data_sha256,
        "k_values": K_VALUES,
        "conditions": CONDITIONS,
        "label_coverage": {category: len(ids) for category, ids in label_ids.items()},
        "author_coverage": {category: len(ids) for category, ids in author_ids.items()},
        "run_summary": summary_rows,
        "category_summary": category_summary,
        "passage_scores": scores.to_dict(orient="records"),
        "headline": (
            f"selectivity k=1 jac={selectivity('jacobian', 1):.3f} "
            f"k=10 jac={selectivity('jacobian', 10):.3f} "
            f"logit k=1={selectivity('logit', 1):.3f}"
        ),
    }
    return payload, {"_trial_scores.csv.gz": scores}


if __name__ == "__main__":
    raise SystemExit(common.run_experiment(EXPERIMENT_NAME, run_one))
