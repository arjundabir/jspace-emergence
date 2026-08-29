from __future__ import annotations

import itertools
import random
from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch

from experiments import common
from experiments.common import ActivationRecorder, LensRun

EXPERIMENT_NAME = "ignition"
PAIR_KINDS = ["country", "alt", "idiom", "scrambled"]
N_COUNTRY_PAIRS = 16
N_CARRIERS = 8
N_ALPHAS = 21
N_DISTRACTORS = 4
MAJORITY_MARGIN = 0.25


def build_pairs(data: dict, rng: random.Random, n_country_pairs: int) -> dict:
    country_pairs = list(itertools.combinations(data["countries_12"], 2))
    if n_country_pairs < len(country_pairs):
        country_pairs = rng.sample(country_pairs, n_country_pairs)
    return {
        "country": [tuple(pair) for pair in country_pairs],
        "alt": [("France", word) for word in data["alt_words"]],
        "idiom": [tuple(pair) for pair in data["idiom_pairs"]],
        "scrambled": [tuple(pair) for pair in data["scrambled_pairs"]],
    }


def build_distractor_pools(data: dict) -> dict[str, list[str]]:
    """Candidate negative-control words per pair kind — the vocabulary that
    kind's pairs are themselves drawn from, so a distractor is always the
    same sort of word as the target it stands in for."""
    return {
        "country": list(data["countries_12"]),
        "alt": ["France"] + list(data["alt_words"]),
        "idiom": [word for pair in data["idiom_pairs"] for word in pair],
        "scrambled": [word for pair in data["scrambled_pairs"] for word in pair],
    }


def matched_distractors(
    tokenizer, pool: Sequence[str], word_a: str, word_b: str, n_distractors: int
) -> list[tuple[str, int]]:
    """``(word, token_id)`` for the pool words nearest the pair in merge rank.

    Selection depends only on the pair and the tokenizer — never on a rank
    that was read. Candidates must have a single-token leading-space form and
    must not be in the mixture; the nearest token ids keep the control
    approximately frequency-matched to the target.
    """
    excluded = {word_a, word_b}
    candidates: list[tuple[str, int]] = []
    for word in dict.fromkeys(pool):  # de-duplicate, keep dataset order
        if word in excluded:
            continue
        ids = tokenizer.encode(" " + word, add_special_tokens=False)
        if len(ids) == 1:
            candidates.append((word, ids[0]))
    ids_a = tokenizer.encode(" " + word_a, add_special_tokens=False)
    ids_b = tokenizer.encode(" " + word_b, add_special_tokens=False)
    anchors = [ids[0] for ids in (ids_a, ids_b) if len(ids) == 1]
    if not anchors:
        return []
    candidates.sort(
        key=lambda item: (
            sum(abs(item[1] - anchor) for anchor in anchors),
            item[0],  # deterministic tie-break
        )
    )
    return candidates[:n_distractors]


def crossing_alpha(alphas: np.ndarray, shares: np.ndarray, level: float) -> float:
    """First upward crossing of ``level`` on the share's monotone envelope.

    NaN when the curve starts at/above the level or never reaches it: no
    crossing is observed inside the sweep. Returning 0 for a curve that is
    high everywhere would fabricate a maximally sharp "transition" in exactly
    the no-transition regime the scrambled controls expose.
    """
    envelope = np.maximum.accumulate(shares)
    if envelope[0] >= level:
        return float("nan")
    above = np.nonzero(envelope >= level)[0]
    if not len(above):
        return float("nan")
    j = int(above[0])
    lower, upper = envelope[j - 1], envelope[j]
    if upper == lower:
        return float(alphas[j])
    t = (level - lower) / (upper - lower)
    return float(alphas[j - 1] + t * (alphas[j] - alphas[j - 1]))


def tidy_target_ranks(
    wide: pd.DataFrame,
    distractors_by_pair: dict[tuple[str, str], list[str]],
    majority_margin: float,
) -> pd.DataFrame:
    """Long form of the band-min ranks: one row per scored word per trial.

    ``role`` is ``"a"``/``"b"`` for the mixture members and ``"distractor"``
    for the matched controls; ``majority`` names which member the mixture
    favours (``"none"`` inside ``majority_margin`` of an even mix).
    """
    if wide.empty:
        return wide
    keys = ["method", "kind", "pair", "carrier_index", "alpha"]
    distractor_columns = sorted(
        column for column in wide.columns if column.startswith("rank_distractor")
    )
    frames = []
    for role, column in [("a", "rank_a_bandmin"), ("b", "rank_b_bandmin")]:
        part = wide[keys + [column, "word_a", "word_b"]].copy()
        part["role"] = role
        part["word"] = part["word_a"] if role == "a" else part["word_b"]
        part["rank_bandmin"] = part[column]
        frames.append(part[keys + ["role", "word", "rank_bandmin"]])
    for index, column in enumerate(distractor_columns):
        part = wide[keys + [column]].copy()
        part["role"] = "distractor"
        part["word"] = [
            (distractors_by_pair.get((kind, pair), [None] * (index + 1)) + [None])[index]
            for kind, pair in zip(part["kind"], part["pair"])
        ]
        part["rank_bandmin"] = part[column]
        frames.append(part[keys + ["role", "word", "rank_bandmin"]])
    long = pd.concat(frames, ignore_index=True)
    long = long[long["word"].notna()]
    margin = long["alpha"] - 0.5
    long["majority"] = np.where(
        margin >= majority_margin, "a", np.where(margin <= -majority_margin, "b", "none")
    )
    return long.sort_values(keys + ["role", "word"]).reset_index(drop=True)


def specificity_rows(targets: pd.DataFrame) -> list[dict]:
    """Per (method, kind) target-specificity summary from the long ranks."""
    if targets.empty:
        return []
    scored = targets[targets["majority"] != "none"]
    correct = scored[scored["role"] == scored["majority"]]
    keys = ["method", "kind", "pair", "carrier_index", "alpha"]
    control = scored[scored["role"] == "distractor"]
    merged = control.merge(
        correct[keys + ["rank_bandmin"]], on=keys, suffixes=("_control", "_correct")
    )
    merged["log2_ratio"] = np.log2(
        merged["rank_bandmin_control"] / merged["rank_bandmin_correct"]
    )
    out = []
    for (method, kind), group in merged.groupby(["method", "kind"], sort=False):
        finite = group["log2_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
        out.append({
            "method": method,
            "kind": kind,
            "n_specificity_pairs": int(len(finite)),
            "specificity_log2_median": float(finite.median()) if len(finite) else float("nan"),
            "specificity_log2_mean": float(finite.mean()) if len(finite) else float("nan"),
            "median_rank_correct": float(group["rank_bandmin_correct"].median()),
            "median_rank_distractor": float(group["rank_bandmin_control"].median()),
        })
    return out


@torch.no_grad()
def run_one(run: LensRun) -> tuple[dict, dict[str, pd.DataFrame]]:
    data_path, data, data_sha256 = common.load_experiment_data(EXPERIMENT_NAME)
    alphas = np.linspace(0.0, 1.0, N_ALPHAS)
    alphas_tensor = torch.tensor(alphas, dtype=torch.float32)

    pairs_by_kind = build_pairs(data, random.Random(common.RANDOM_SEED), N_COUNTRY_PAIRS)
    templates_by_kind = {
        "country": data["ctx_templates"],
        "alt": data["noun_ctx_templates"],
        "idiom": data["noun_ctx_templates"],
        "scrambled": data["noun_ctx_templates"],
    }

    embed = run.lens_model._embed_tokens
    reference_layer = run.band[-1]
    band_layer_set = set(run.band)
    distractor_pools = build_distractor_pools(data)

    rows = []
    target_rows: list[dict] = []
    distractors_by_pair: dict[tuple[str, str], list[str]] = {}
    skipped_pairs: dict[str, int] = {kind: 0 for kind in PAIR_KINDS}
    skipped_carriers: dict[str, int] = {kind: 0 for kind in PAIR_KINDS}
    for kind_index, kind in enumerate(PAIR_KINDS):
        templates = templates_by_kind[kind]
        rng = random.Random(common.RANDOM_SEED * 1_000_003 + kind_index)
        carrier_indices = rng.sample(range(len(templates)), min(N_CARRIERS, len(templates)))
        for word_a, word_b in pairs_by_kind[kind]:
            ids_a = run.tokenizer.encode(" " + word_a, add_special_tokens=False)
            ids_b = run.tokenizer.encode(" " + word_b, add_special_tokens=False)
            if len(ids_a) != 1 or len(ids_b) != 1:
                skipped_pairs[kind] += 1
                continue
            token_a, token_b = ids_a[0], ids_b[0]
            pair_name = f"{word_a}|{word_b}"
            distractors = matched_distractors(
                run.tokenizer, distractor_pools[kind], word_a, word_b, N_DISTRACTORS
            )
            distractors_by_pair[(kind, pair_name)] = [word for word, _ in distractors]

            for carrier_index in carrier_indices:
                template = templates[carrier_index]
                # Tokenize the fully formed sentence and locate {W} by char
                # offsets. The word must occupy the single leading-space token
                # whose lens vector is tracked; otherwise this carrier is
                # unmeasurable under the tokenizer and is skipped.
                prefix_len = len(template.split("{W}")[0])
                full_text = template.replace("{W}", word_a)
                input_ids, offsets = common.encode_with_offsets(run.tokenizer, full_text)
                span = [
                    index
                    for index, (start, end) in enumerate(offsets)
                    if start < prefix_len + len(word_a) and end > prefix_len and end > start
                ]
                if len(span) != 1 or input_ids[span[0]] != token_a:
                    skipped_carriers[kind] += 1
                    continue
                w_position = span[0]

                ids_tensor = torch.tensor(
                    [input_ids], dtype=torch.long, device=run.lens_model.input_device
                )
                base_embeds = embed(ids_tensor)  # [1, seq, d_model]
                emb_a = embed(torch.tensor([token_a], device=run.lens_model.input_device))[0].float()
                emb_b = embed(torch.tensor([token_b], device=run.lens_model.input_device))[0].float()
                batch = base_embeds.repeat(len(alphas), 1, 1).clone()
                mix = (
                    alphas_tensor.to(run.lens_model.input_device).unsqueeze(1)
                    * emb_a.unsqueeze(0)
                    + (1.0 - alphas_tensor.to(run.lens_model.input_device)).unsqueeze(1)
                    * emb_b.unsqueeze(0)
                )
                batch[:, w_position, :] = mix.to(batch.dtype)

                with ActivationRecorder(run.lens_model.layers, at=run.all_layers) as recorder:
                    run.lens_model._text_module(inputs_embeds=batch, use_cache=False)
                    residuals = {
                        layer: recorder.activations[layer][:, w_position].detach().float()
                        for layer in run.all_layers
                    }
                del ids_tensor, base_embeds, batch

                tracked_ids = [token_a, token_b] + [token_id for _, token_id in distractors]
                for method in common.METHODS:
                    # Band-minimum rank per tracked word: one number per
                    # (trial, alpha) comparing target and matched controls.
                    band_min = np.full((len(alphas), len(tracked_ids)), np.inf)
                    for layer in run.all_layers:
                        logits = run.lens_logits(residuals[layer], layer, method)
                        ranks = common.ranks_of(logits, tracked_ids).numpy()
                        if layer in band_layer_set:
                            band_min = np.minimum(band_min, ranks)
                        share = (1.0 / ranks[:, 0]) / (1.0 / ranks[:, 0] + 1.0 / ranks[:, 1])
                        for alpha_index, alpha in enumerate(alphas):
                            rows.append({
                                "method": method,
                                "kind": kind,
                                "pair": pair_name,
                                "carrier_index": carrier_index,
                                "layer": layer,
                                "alpha": round(float(alpha), 4),
                                "rank_a": int(ranks[alpha_index, 0]),
                                "rank_b": int(ranks[alpha_index, 1]),
                                "share_a": round(float(share[alpha_index]), 5),
                            })
                        del logits

                    for alpha_index, alpha in enumerate(alphas):
                        target_rows.append({
                            "method": method,
                            "kind": kind,
                            "pair": pair_name,
                            "carrier_index": carrier_index,
                            "alpha": round(float(alpha), 4),
                            "word_a": word_a,
                            "word_b": word_b,
                            "rank_a_bandmin": float(band_min[alpha_index, 0]),
                            "rank_b_bandmin": float(band_min[alpha_index, 1]),
                            **{
                                f"rank_distractor{index}_bandmin": float(
                                    band_min[alpha_index, 2 + index]
                                )
                                for index in range(len(distractors))
                            },
                        })
                del residuals
            if run.device.type == "cuda":
                torch.cuda.empty_cache()

    targets = tidy_target_ranks(pd.DataFrame(target_rows), distractors_by_pair, MAJORITY_MARGIN)
    scores = pd.DataFrame(rows)

    # Per-trial thresholds (reference layer) and per-layer widths.
    trial_rows = []
    layer_rows = []
    grouped = scores.groupby(["method", "kind", "pair", "carrier_index"], sort=False)
    for (method, kind, pair, carrier_index), trial in grouped:
        by_layer = {
            layer: layer_frame.sort_values("alpha")["share_a"].to_numpy()
            for layer, layer_frame in trial.groupby("layer")
        }
        reference_shares = by_layer[reference_layer]
        threshold = crossing_alpha(alphas, reference_shares, 0.5)
        trial_rows.append({
            "method": method,
            "kind": kind,
            "pair": pair,
            "carrier_index": carrier_index,
            "threshold_alpha": threshold,
            "share_alpha0": float(reference_shares[0]),
            "share_alpha1": float(reference_shares[-1]),
        })
        threshold_index = (
            int(np.argmin(np.abs(alphas - threshold))) if np.isfinite(threshold) else None
        )
        for layer, shares in by_layer.items():
            lo = crossing_alpha(alphas, shares, 0.1)
            hi = crossing_alpha(alphas, shares, 0.9)
            if threshold_index is None:
                # No threshold observed: unmeasurable, not "non-bimodal".
                share_at_threshold = float("nan")
                bimodal = float("nan")
            else:
                share_at_threshold = float(shares[threshold_index])
                bimodal = float(share_at_threshold <= 0.2 or share_at_threshold >= 0.8)
            layer_rows.append({
                "method": method,
                "kind": kind,
                "pair": pair,
                "carrier_index": carrier_index,
                "layer": layer,
                "transition_width": hi - lo,
                "share_at_threshold": share_at_threshold,
                "bimodal_at_threshold": bimodal,
            })

    trials = pd.DataFrame(trial_rows)
    layer_stats = pd.DataFrame(layer_rows)

    # Medians and fractions skip NaN (unmeasurable trials); measured counts
    # are reported alongside so censoring is visible, not silent.
    layer_profile = [
        {
            "method": method,
            "kind": kind,
            "layer": int(layer),
            "layer_pct": 100.0 * layer / max(run.lens_model.n_layers - 1, 1),
            "n_trials": len(group),
            "n_width_measured": int(group["transition_width"].notna().sum()),
            "median_transition_width": float(group["transition_width"].median()),
            "n_bimodal_measured": int(group["bimodal_at_threshold"].notna().sum()),
            "bimodal_fraction_at_threshold": float(group["bimodal_at_threshold"].mean()),
        }
        for (method, kind, layer), group in layer_stats.groupby(
            ["method", "kind", "layer"], sort=False
        )
    ]

    def layer_nearest(fraction: float) -> int:
        target = fraction * (run.lens_model.n_layers - 1)
        return min(run.all_layers, key=lambda layer: abs(layer - target))

    depth_probes = {
        "early": layer_nearest(0.25),
        "mid": layer_nearest(0.50),
        "late": layer_nearest(0.75),
    }

    summary_rows = []
    for (method, kind), group in layer_stats.groupby(["method", "kind"], sort=False):
        trial_group = trials[(trials["method"] == method) & (trials["kind"] == kind)]
        entry = {
            "method": method,
            "kind": kind,
            "n_trials": len(trial_group),
            "n_pairs_skipped": skipped_pairs[kind],
            "n_carrier_trials_skipped": skipped_carriers[kind],
            "threshold_defined_fraction": float(trial_group["threshold_alpha"].notna().mean()),
            "mean_threshold_alpha": float(trial_group["threshold_alpha"].mean()),
        }
        for label, layer in depth_probes.items():
            at_layer = group[group["layer"] == layer]
            entry[f"width_{label}_median"] = float(at_layer["transition_width"].median())
            entry[f"width_{label}_n"] = int(at_layer["transition_width"].notna().sum())
            entry[f"bimodal_{label}_fraction"] = float(at_layer["bimodal_at_threshold"].mean())
            entry[f"bimodal_{label}_n"] = int(at_layer["bimodal_at_threshold"].notna().sum())
        summary_rows.append(entry)

    specificity_by_key = {
        (row["method"], row["kind"]): row for row in specificity_rows(targets)
    }
    for entry in summary_rows:
        extra = specificity_by_key.get((entry["method"], entry["kind"]), {})
        for column in (
            "n_specificity_pairs",
            "specificity_log2_median",
            "specificity_log2_mean",
            "median_rank_correct",
            "median_rank_distractor",
        ):
            entry[column] = extra.get(column, float("nan"))

    def headline_value(kind: str, column: str) -> float:
        match = [
            row[column]
            for row in summary_rows
            if row["method"] == "jacobian" and row["kind"] == kind
        ]
        return match[0] if match else float("nan")

    payload = {
        "dataset_sha256": data_sha256,
        "alpha_grid": [round(float(a), 4) for a in alphas],
        "pair_kinds": PAIR_KINDS,
        "n_pairs_by_kind": {k: len(v) for k, v in pairs_by_kind.items()},
        "n_pairs_skipped": skipped_pairs,
        "n_carrier_trials_skipped": skipped_carriers,
        "n_carriers": N_CARRIERS,
        "reference_layer": reference_layer,
        "band_layers": list(run.band),
        "n_distractors": N_DISTRACTORS,
        "majority_margin": MAJORITY_MARGIN,
        "distractors_by_pair": {
            f"{kind}::{pair}": words for (kind, pair), words in distractors_by_pair.items()
        },
        "depth_probe_layers": depth_probes,
        "run_summary": summary_rows,
        "layer_profile": layer_profile,
        "trial_thresholds": trials.to_dict(orient="records"),
        "headline": (
            f"country width mid={headline_value('country', 'width_mid_median'):.3f} "
            f"late={headline_value('country', 'width_late_median'):.3f} "
            f"bimodal_late={headline_value('country', 'bimodal_late_fraction'):.2f}"
        ),
    }
    return payload, {
        "_share_curves.csv.gz": scores,
        "_layer_widths.csv.gz": layer_stats,
        "_target_ranks.csv.gz": targets,
    }


if __name__ == "__main__":
    raise SystemExit(common.run_experiment(EXPERIMENT_NAME, run_one))
