from __future__ import annotations

import random

import numpy as np
import pandas as pd
import torch

from experiments import common
from experiments.common import LensRun

EXPERIMENT_NAME = "capacity"
K_VALUES = [1, 5, 10, 25, 50]
WORDS_PER_BLOCK = 20
TRIALS = 8
TOP_K = 25
MIN_LOAD = 20


def build_canon(tokenizer, pools: list[dict], targets_per_family: dict) -> dict:
    canon: dict[str, list[str]] = {}
    for pool in pools:
        name = pool["name"]
        budget = int(targets_per_family[name])
        survivors = []
        for word in pool["pool"]:
            ids = tokenizer.encode(" " + word, add_special_tokens=False)
            if len(ids) == 1:
                survivors.append(word)
            if len(survivors) >= budget:
                break
        canon[name] = survivors
    return canon


def build_trial(rng, canon: dict, block_families: list[str]) -> tuple[list[str], list[str], list[int]]:
    order = list(block_families)
    rng.shuffle(order)
    words: list[str] = []
    families: list[str] = []
    block_indices: list[int] = []
    for block_index, family in enumerate(order):
        pool = canon[family]
        count = min(WORDS_PER_BLOCK, len(pool))
        for word in rng.sample(pool, count):
            words.append(word)
            families.append(family)
            block_indices.append(block_index)
    return words, families, block_indices


def match_lures(tokenizer, canon: dict, words: list[str], families: list[str]) -> dict:
    def leading_space_id(word: str) -> int | None:
        ids = tokenizer.encode(" " + word, add_special_tokens=False)
        return ids[0] if len(ids) == 1 else None

    presented_by_family: dict[str, list[str]] = {}
    for word, family in zip(words, families):
        presented_by_family.setdefault(family, []).append(word)

    lures: dict[str, str] = {}
    for family in sorted(presented_by_family):
        presented = set(presented_by_family[family])
        available = []
        for candidate in canon[family]:
            if candidate in presented:
                continue
            token_id = leading_space_id(candidate)
            if token_id is not None:
                available.append((candidate, token_id))
        used: set[str] = set()
        ordered = sorted(
            ((word, leading_space_id(word)) for word in presented_by_family[family]),
            key=lambda item: (item[1] is None, item[1] or 0, item[0]),
        )
        for word, token_id in ordered:
            if token_id is None:
                continue
            remaining = [item for item in available if item[0] not in used]
            if not remaining:
                break
            lure = min(remaining, key=lambda item: (abs(item[1] - token_id), item[0]))[0]
            used.add(lure)
            lures[word] = lure
    return lures


def list_text_and_comma_chars(words: list[str]) -> tuple[str, list[int]]:
    text = ""
    comma_chars: list[int] = []
    for index, word in enumerate(words):
        if index:
            text += " "
        text += word
        comma_chars.append(len(text))
        text += ","
    return text, comma_chars


def comma_token_positions(offsets: list[tuple[int, int]], comma_chars: list[int]) -> list[int]:
    positions = []
    for char_index in comma_chars:
        position = next(
            index
            for index, (start, end) in enumerate(offsets)
            if start <= char_index < end
        )
        positions.append(position)
    return positions


def retention_specificity(
    words_df: pd.DataFrame, lures_df: pd.DataFrame, min_load: int
) -> dict[str, dict]:
    seen = words_df[words_df["read"] & (words_df["comma_index"] >= min_load)]
    keys = ["trial", "method", "comma_index"]
    paired = seen.merge(
        lures_df[keys + ["lure_for", "rank_bandmin"]],
        left_on=keys + ["word"],
        right_on=keys + ["lure_for"],
        suffixes=("_seen", "_lure"),
    )
    paired = paired.assign(
        log2_ratio=np.log2(paired["rank_bandmin_lure"] / paired["rank_bandmin_seen"])
    )
    out: dict[str, dict] = {}
    for method, group in paired.groupby("method", sort=False):
        finite = group["log2_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
        out[method] = {
            "n_retention_pairs": int(len(finite)),
            "retention_specificity_log2_median": (
                float(finite.median()) if len(finite) else float("nan")
            ),
            "retention_specificity_log2_mean": (
                float(finite.mean()) if len(finite) else float("nan")
            ),
            "median_rank_seen": float(group["rank_bandmin_seen"].median()),
            "median_rank_lure": float(group["rank_bandmin_lure"].median()),
        }
    return out


@torch.no_grad()
def run_one(run: LensRun) -> tuple[dict, dict[str, pd.DataFrame]]:
    data_path, data, data_sha256 = common.load_experiment_data(EXPERIMENT_NAME)
    block_families = data["block_families"]
    canon = build_canon(run.tokenizer, data["candidate_pools"], data["targets_per_family"])
    proto_words = {pool["name"]: pool["proto"] for pool in data["candidate_pools"]}

    # Tracked vocabulary is trial-invariant: single-token surface forms for
    # every canon word and each family's proto (category-label) words.
    canon_form_ids = {
        word: list(common.single_token_forms(run.tokenizer, word).values())
        for family_words in canon.values()
        for word in family_words
    }
    proto_ids_by_family = {
        family: sorted({
            token_id
            for label in labels
            for token_id in common.single_token_forms(run.tokenizer, label).values()
        })
        for family, labels in proto_words.items()
    }

    rows = []
    for trial in range(TRIALS):
        rng = random.Random(common.RANDOM_SEED * 100_003 + trial)
        words, families, block_indices = build_trial(rng, canon, block_families)
        text, comma_chars = list_text_and_comma_chars(words)
        input_ids, offsets = common.encode_with_offsets(run.tokenizer, text)
        comma_positions = comma_token_positions(offsets, comma_chars)

        lure_of = match_lures(run.tokenizer, canon, words, families)
        lure_words = [lure_of[word] for word in words if word in lure_of]
        family_of_lure = {
            lure_of[word]: families[index]
            for index, word in enumerate(words)
            if word in lure_of
        }
        block_of_lure = {
            lure_of[word]: block_indices[index]
            for index, word in enumerate(words)
            if word in lure_of
        }
        form_ids_by_word = {word: canon_form_ids[word] for word in words + lure_words}
        tracked_ids = sorted(
            {token_id for ids in form_ids_by_word.values() for token_id in ids}
            | {token_id for ids in proto_ids_by_family.values() for token_id in ids}
        )
        column_of = {token_id: column for column, token_id in enumerate(tracked_ids)}

        residuals = common.capture_band_residuals(run, input_ids, comma_positions)

        for method in common.METHODS:
            band_min = np.full((len(comma_positions), len(tracked_ids)), np.inf)
            for layer in run.band:
                logits = run.lens_logits(residuals[layer], layer, method)
                layer_ranks = common.ranks_of(logits, tracked_ids).numpy()
                band_min = np.minimum(band_min, layer_ranks)
                del logits

            for comma_index in range(len(comma_positions)):
                for word_index, word in enumerate(words):
                    columns = [column_of[i] for i in form_ids_by_word[word]]
                    rank = float(band_min[comma_index, columns].min()) if columns else float("inf")
                    rows.append({
                        "trial": trial,
                        "method": method,
                        "comma_index": comma_index,
                        "word_index": word_index,
                        "word": word,
                        "family": families[word_index],
                        "block_index": block_indices[word_index],
                        "comma_block_index": block_indices[comma_index],
                        "is_proto": False,
                        "role": "list",
                        "lure_for": "",
                        "read": word_index <= comma_index,
                        "rank_bandmin": rank,
                    })
                # Matched never-presented control for each list word, read at
                # the same comma.
                for word_index, word in enumerate(words):
                    lure = lure_of.get(word)
                    if lure is None:
                        continue
                    columns = [column_of[i] for i in form_ids_by_word[lure]]
                    rank = float(band_min[comma_index, columns].min()) if columns else float("inf")
                    rows.append({
                        "trial": trial,
                        "method": method,
                        "comma_index": comma_index,
                        "word_index": word_index,
                        "word": lure,
                        "family": family_of_lure[lure],
                        "block_index": block_of_lure[lure],
                        "comma_block_index": block_indices[comma_index],
                        "is_proto": False,
                        "role": "lure",
                        "lure_for": word,
                        "read": False,
                        "rank_bandmin": rank,
                    })
                for family, ids in proto_ids_by_family.items():
                    columns = [column_of[i] for i in ids]
                    rank = float(band_min[comma_index, columns].min()) if columns else float("inf")
                    rows.append({
                        "trial": trial,
                        "method": method,
                        "comma_index": comma_index,
                        "word_index": -1,
                        "word": f"<proto:{family}>",
                        "family": family,
                        "block_index": -1,
                        "comma_block_index": block_indices[comma_index],
                        "is_proto": True,
                        "role": "proto",
                        "lure_for": "",
                        "read": False,
                        "rank_bandmin": rank,
                    })
        del residuals
        if run.device.type == "cuda":
            torch.cuda.empty_cache()

    scores = pd.DataFrame(rows)
    # Every presented-word metric holds the lures out as a parallel control.
    words_df = scores[scores["role"] == "list"]
    lures_df = scores[scores["role"] == "lure"]
    proto_df = scores[scores["role"] == "proto"]

    # Comma-position profile: mean read words present at rank <= k.
    comma_profile = []
    for method in common.METHODS:
        method_words = words_df[words_df["method"] == method]
        read = method_words[method_words["read"]]
        for comma_index, group in read.groupby("comma_index"):
            per_trial = group.groupby("trial")
            entry = {"method": method, "comma_index": int(comma_index)}
            for k in K_VALUES:
                counts = per_trial["rank_bandmin"].apply(lambda r, k=k: (r <= k).sum())
                entry[f"read_in_top{k}_mean"] = float(counts.mean())
            comma_profile.append(entry)

    def hit_counts(frame: pd.DataFrame, k: int, index: pd.MultiIndex) -> pd.Series:
        counts = (
            frame.assign(hit=frame["rank_bandmin"] <= k)
            .groupby(["trial", "comma_index"])["hit"]
            .sum()
        )
        return counts.reindex(index, fill_value=0)

    summary_rows = []
    for method in common.METHODS:
        method_words = words_df[words_df["method"] == method]
        all_commas = pd.MultiIndex.from_frame(
            method_words[["trial", "comma_index"]].drop_duplicates()
        )
        read_mask = method_words["read"]
        current = method_words["block_index"] == method_words["comma_block_index"]

        read_counts = hit_counts(method_words[read_mask], TOP_K, all_commas)
        current_read_counts = hit_counts(method_words[read_mask & current], TOP_K, all_commas)
        # Pre-activation: unread words of the block currently being read.
        unread_current_counts = hit_counts(method_words[~read_mask & current], TOP_K, all_commas)

        # Retention: previous block's words, measured 5 words after each block
        # switch; switch positions derive per trial from the block structure.
        retention_samples = []
        comma_blocks = method_words.drop_duplicates(["trial", "comma_index"])
        for trial, trial_blocks in comma_blocks.groupby("trial"):
            trial_words = method_words[method_words["trial"] == trial]
            for block in sorted(trial_blocks["comma_block_index"].unique()):
                if block == 0:
                    continue
                first = int(
                    trial_blocks.loc[
                        trial_blocks["comma_block_index"] == block, "comma_index"
                    ].min()
                )
                at_switch = trial_words[
                    (trial_words["comma_index"] == first + 4)
                    & (trial_words["block_index"] == block - 1)
                ]
                if len(at_switch):
                    retention_samples.append(float((at_switch["rank_bandmin"] <= TOP_K).mean()))
        retention = float(np.mean(retention_samples)) if retention_samples else float("nan")

        # Family label of the block being read, resolved per trial.
        proto_hits = []
        for trial, group in proto_df[proto_df["method"] == method].groupby("trial"):
            trial_words = words_df[
                (words_df["trial"] == trial) & (words_df["method"] == method)
            ]
            family_of_block = (
                trial_words.drop_duplicates("block_index")
                .set_index("block_index")["family"]
                .to_dict()
            )
            current_family = group["comma_block_index"].map(family_of_block)
            matched = group[group["family"] == current_family]
            proto_hits.append((matched["rank_bandmin"] <= TOP_K).mean())

        final_comma = method_words["comma_index"].max()
        entry = {
            "method": method,
            "k": TOP_K,
            "n_trials": TRIALS,
            "n_words_per_list": int(method_words["word_index"].max()) + 1,
            f"read_in_top{TOP_K}_mean": float(read_counts.mean()),
            f"read_in_top{TOP_K}_final": float(
                read_counts[
                    read_counts.index.get_level_values("comma_index") == final_comma
                ].mean()
            ),
            "current_block_read_mean": float(current_read_counts.mean()),
            "current_block_unread_mean": float(unread_current_counts.mean()),
            "prev_block_retention_after_5": float(retention),
            "proto_label_hit_rate": float(np.nanmean(proto_hits)),
        }
        for k in K_VALUES:
            entry[f"read_in_top{k}_mean_allk"] = float(
                hit_counts(method_words[read_mask], k, all_commas).mean()
            )
        summary_rows.append(entry)

    specificity = retention_specificity(words_df, lures_df, MIN_LOAD)
    for entry in summary_rows:
        entry.update(
            specificity.get(
                entry["method"],
                {
                    "n_retention_pairs": 0,
                    "retention_specificity_log2_median": float("nan"),
                    "retention_specificity_log2_mean": float("nan"),
                    "median_rank_seen": float("nan"),
                    "median_rank_lure": float("nan"),
                },
            )
        )

    headline = next(row for row in summary_rows if row["method"] == "jacobian")
    payload = {
        "dataset_sha256": data_sha256,
        "block_families": block_families,
        "canon_sizes": {family: len(words) for family, words in canon.items()},
        "words_per_block": WORDS_PER_BLOCK,
        "k_values": K_VALUES,
        "top_k": TOP_K,
        "min_load": MIN_LOAD,
        "band_layers": list(run.band),
        "n_trials": TRIALS,
        "run_summary": summary_rows,
        "comma_profile": comma_profile,
        "headline": (
            f"read@{TOP_K} mean={headline[f'read_in_top{TOP_K}_mean']:.2f} "
            f"retention5={headline['prev_block_retention_after_5']:.3f} "
            f"S_ret={headline['retention_specificity_log2_median']:.2f}"
        ),
    }
    return payload, {"_word_ranks.csv.gz": scores}


if __name__ == "__main__":
    raise SystemExit(common.run_experiment(EXPERIMENT_NAME, run_one))
