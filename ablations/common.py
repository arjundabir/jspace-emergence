"""Shared machinery for the six J-lens ablations.

Each ablation projects the first intermediate's normalized J-lens direction
out of the residual stream at every prompt position, across the fitted layers
inside the workspace band, then compares clean vs ablated next-token
distributions at the task's readout position: KL(clean || ablated), and the
clean vs ablated rank and log-probability of the scored answer. The logit-lens
direction (J = I) is the baseline method.

A task supplies its dataset slug and an example-preparation function;
whole-answer tasks (poetry, multilingual) additionally score the full target
span by teacher-forced length-normalized mean log-probability, so multi-token
targets stay usable. Two combined CSVs per model are written under
``results/ablations/<model>/``: ``<task>_ablation_summary.csv`` (one row per
training step and method) and ``<task>_ablation_per_example.csv``.
"""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
import torch

import jlens

from lens.load_lens import FITS_ROOT, load_lens, lens_path_for, pythia_layout
from lens.load_models import CHECKPOINT_STEPS, REPO_ROOT, load_model

DATA_ROOT = REPO_ROOT / "jacobian-lens" / "data" / "evaluations"
RESULTS_ROOT = REPO_ROOT / "results" / "ablations"

#: Workspace band as a fraction of depth, fixed across models and checkpoints
#: so the emergence claim compares the same band at every step. Located on
#: Pythia by the excess-kurtosis minimum of the fitted lens, which sits inside
#: (0.35, 0.90) at the last three checkpoints of every sizeable model.
BAND_FRACTION = (0.35, 0.90)


def band_layers(all_layers: list[int], n_layers: int) -> list[int]:
    """Fitted layers inside the fractional workspace band.

    Falls back to every fitted layer when the band selects none, which is
    what happens on shallow models.
    """
    last = max(n_layers - 1, 1)
    low, high = BAND_FRACTION
    inside = [layer for layer in all_layers if low <= layer / last <= high]
    return inside or list(all_layers)


def surface_forms(word: str) -> list[str]:
    return list(dict.fromkeys([word, " " + word]))


def first_single_token_id(tokenizer, word: str) -> tuple[int, str]:
    for surface in surface_forms(word):
        ids = tokenizer.encode(surface, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0], surface
    raise ValueError(f"No single-token form for {word!r}")


def try_single_token_id(tokenizer, word: str) -> tuple[int, str] | None:
    try:
        return first_single_token_id(tokenizer, word)
    except ValueError:
        return None


def one_based_rank(logits: torch.Tensor, token_id: int) -> int:
    return int((logits > logits[int(token_id)]).sum().item()) + 1


def best_rank_logprob(tokenizer, logits: torch.Tensor, word: str):
    """Best (rank, log-prob, surface) over single-token surfaces of ``word``."""
    log_probs = logits.float().log_softmax(dim=-1)
    best = (float("inf"), float("nan"), None)
    for surface in surface_forms(word):
        ids = tokenizer.encode(surface, add_special_tokens=False)
        if len(ids) != 1:
            continue
        tid = ids[0]
        rank = one_based_rank(logits, tid)
        if rank < best[0]:
            best = (float(rank), float(log_probs[tid]), surface)
    if best[2] is None:
        raise ValueError(f"No single-token form for {word!r}")
    return best


def teacher_forced_mean_logprob(
    logits: torch.Tensor, input_ids_list: list[int], target_positions: list[int]
) -> float:
    """Length-normalized mean log-prob of the target tokens under teacher forcing.

    For each target index ``pos``, score ``input_ids[pos]`` under
    ``logits[pos - 1]``.
    """
    total = 0.0
    for pos in target_positions:
        log_probs = logits[pos - 1].float().log_softmax(dim=-1)
        total += float(log_probs[int(input_ids_list[pos])])
    return total / float(len(target_positions))


def lens_vector(lens, lens_model, layer: int, token_id: int, method: str) -> torch.Tensor:
    """Normalized row ``token_id`` of ``W_U J_layer`` (CPU float32); J = I for logit."""
    unembed_row = lens_model._lm_head.weight[token_id].detach().float().cpu()
    if method == "logit":
        vector = unembed_row.clone()
    else:
        vector = lens.jacobians[layer].float().t() @ unembed_row
    return vector / vector.norm().clamp_min(1e-8)


def make_ablation_hook(vector: torch.Tensor, positions: list[int]):
    def hook(module, inputs, output):
        if torch.is_tensor(output):
            hidden, rest = output, None
        else:
            hidden, rest = output[0], output[1:]
        unit = vector.to(hidden.device, torch.float32)
        hidden = hidden.clone()
        block = hidden[:, positions, :].float()
        block = block - (block @ unit).unsqueeze(-1) * unit
        hidden[:, positions, :] = block.to(hidden.dtype)
        return hidden if rest is None else (hidden, *rest)

    return hook


@torch.no_grad()
def forward_with_hooks(hf_model, lens_model, input_ids, hooks: dict):
    handles = [
        lens_model.layers[layer].register_forward_hook(hook)
        for layer, hook in hooks.items()
    ]
    try:
        return hf_model(input_ids=input_ids).logits[0].float()
    finally:
        for handle in handles:
            handle.remove()


def kl_divergence(clean_logits: torch.Tensor, other_logits: torch.Tensor) -> float:
    log_p = clean_logits.float().log_softmax(dim=-1)
    log_q = other_logits.float().log_softmax(dim=-1)
    return float((log_p.exp() * (log_p - log_q)).sum(dim=-1))


@dataclass(frozen=True)
class Example:
    """One prepared item: what to ablate, where to read out, what to score."""

    input_ids_list: list[int]
    readout_position: int
    ablate_positions: list[int]
    direction_token_id: int
    intermediate: str
    target: str
    target_positions: list[int] | None = None  # whole-answer span, else None
    rank_scorable: bool = True


@dataclass(frozen=True)
class Task:
    """What distinguishes one ablation from the other five."""

    name: str
    slug: str
    prepare_example: Callable[[dict, object], "Example | str"]  # str = skip reason
    whole_answer: bool = False


def prompt_final_prepare(item: dict, tokenizer) -> Example | str:
    """Prompt only; readout = last prompt token; the concept scores itself."""
    concept = item["intermediates"][0]
    try:
        concept_id, _ = first_single_token_id(tokenizer, concept)
    except ValueError as exc:
        return str(exc)
    input_ids_list = tokenizer(
        item["prompt"], return_tensors="pt", truncation=False
    ).input_ids[0].tolist()
    return Example(
        input_ids_list=input_ids_list,
        readout_position=len(input_ids_list) - 1,
        ablate_positions=list(range(len(input_ids_list))),
        direction_token_id=concept_id,
        intermediate=concept,
        target=concept,
    )


def _boundary_encode(item: dict, tokenizer, target: str) -> tuple[list[int], int]:
    """Encode ``prompt + target`` offset-aligned; return ids and the target's first token index."""
    encoded = tokenizer(
        item["prompt"] + target,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=False,
    )
    input_ids_list = encoded.input_ids[0].tolist()
    offsets = [tuple(x) for x in encoded.offset_mapping[0].tolist()]
    boundary = len(item["prompt"])
    target_first = next(i for i, (start, end) in enumerate(offsets) if end > boundary)
    return input_ids_list, target_first


def target_boundary_prepare(item: dict, tokenizer) -> Example | str:
    """Readout at the token before ``target``; both intermediate and target single-token."""
    target = item["target"]
    intermediate = item["intermediates"][0]
    try:
        intermediate_id, _ = first_single_token_id(tokenizer, intermediate)
        first_single_token_id(tokenizer, target)
    except ValueError as exc:
        return str(exc)
    input_ids_list, target_first = _boundary_encode(item, tokenizer, target)
    return Example(
        input_ids_list=input_ids_list,
        readout_position=target_first - 1,
        ablate_positions=list(range(len(input_ids_list))),
        direction_token_id=intermediate_id,
        intermediate=intermediate,
        target=target,
    )


def whole_answer_prepare(item: dict, tokenizer) -> Example | str:
    """Readout before the target; the full target span is scored teacher-forced,
    so a multi-token target stays usable as long as the intermediate is
    single-token. Rank is additionally reported when the target is single-token.
    """
    intermediate = item["intermediates"][0]
    target = item.get("target") or intermediate
    try:
        intermediate_id, _ = first_single_token_id(tokenizer, intermediate)
    except ValueError as exc:
        return str(exc)
    input_ids_list, target_first = _boundary_encode(item, tokenizer, target)
    return Example(
        input_ids_list=input_ids_list,
        readout_position=target_first - 1,
        ablate_positions=list(range(len(input_ids_list))),
        direction_token_id=intermediate_id,
        intermediate=intermediate,
        target=target,
        target_positions=list(range(target_first, len(input_ids_list))),
        rank_scorable=try_single_token_id(tokenizer, target) is not None,
    )
