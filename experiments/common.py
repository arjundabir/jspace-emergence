"""Shared infrastructure for the eleven global-workspace experiment scripts.

The experiments replicate the prompt-set experiments from *Verbalizable
Representations Form a Global Workspace in Language Models*
(<https://transformer-circuits.pub/2026/workspace/index.html>) over every
fitted lens in ``fits/``, writing one JSON artifact per lens under
``results/experiments/<model>/<step>/`` and an
``emergence_summary_<name>.csv`` rebuilt at the end of every run.

Paper protocol conventions implemented here (see
``jacobian-lens/data/experiments/README.md``):

- **Workspace band** — the contiguous mid-network layer range where workspace
  content is read; the fixed fractional band is applied to each lens's fitted
  layers.
- **Methods** — every readout and intervention is scored for the fitted
  Jacobian lens (``method="jacobian"``) and the logit-lens baseline
  (``method="logit"``, i.e. ``J = I``) on identical items.
- **Surface forms** — a word is scored in both its bare and leading-space
  single-token forms and the minimum rank is taken; words with no
  single-token form are excluded from the denominator, not scored as
  failures.
- **Chat framing** — the paper runs on chat-formatted models. The Pythia
  checkpoints are base models, so instruction prompts are emulated as plain
  ``Human: ... \\n\\nAssistant: ...`` text with the response teacher-forced.
  On early checkpoints a null result is expected and is itself the datum.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch

import jlens
from jlens.hooks import ActivationRecorder

from ablations.common import BAND_FRACTION, band_layers
from evals.common import (
    METHODS,
    PARAM_COUNTS,
    RANDOM_SEED,
    TOKENS_PER_STEP,
    discover_lenses,
    encode_with_offsets,
    identity_from_lens_filename,
    json_default,
    mean_stderr,
    one_based_rank,
    output_path_for_lens,
    sha256_file,
    unembed_stable,
)
from lens.load_lens import FITS_ROOT, load_lens, pythia_layout
from lens.load_models import REPO_ROOT, load_model

EXPERIMENT_DATA_DIR = REPO_ROOT / "jacobian-lens" / "data" / "experiments"
RESULTS_ROOT = REPO_ROOT / "results" / "experiments"


def sanitize_for_json(value):
    """Replace non-finite floats with ``None`` recursively.

    ``json.dumps`` would otherwise emit bare ``NaN``/``Infinity`` tokens,
    which strict RFC-8259 parsers reject; ``None`` round-trips to NaN under
    pandas.
    """
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, dict):
        return {key: sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return sanitize_for_json(value.tolist())
    if torch.is_tensor(value):
        return sanitize_for_json(value.detach().cpu().tolist())
    return value


_ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
]
_TENS = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
    "eighty", "ninety",
]


def number_word(n: int) -> str:
    """English word for a small non-negative integer (0-99, hyphenless tens)."""
    if not 0 <= n < 100:
        raise ValueError(f"number_word supports 0-99, got {n}")
    if n < 20:
        return _ONES[n]
    tens, ones = divmod(n, 10)
    return _TENS[tens] if ones == 0 else f"{_TENS[tens]}{_ONES[ones]}"


def ranks_of(logits: torch.Tensor, token_ids: Sequence[int]) -> torch.Tensor:
    """One-based ranks of ``token_ids`` in each row of ``logits``.

    One sort per row plus a batched ``searchsorted``: ``[n_positions, vocab]``
    logits in, ``[n_positions, len(token_ids)]`` int64 ranks on CPU out, with
    the same strict-greater tie convention as ``one_based_rank``.
    """
    ids = torch.as_tensor(list(token_ids), dtype=torch.long, device=logits.device)
    gathered = logits.index_select(1, ids)
    # Ascending sort of -logits; the left insertion point of -x counts the
    # entries with v > x: exactly the strict-greater rank.
    neg_sorted = torch.sort(-logits, dim=1).values
    ranks = torch.searchsorted(neg_sorted, -gathered, right=False) + 1
    return ranks.to(dtype=torch.int64).cpu()


def teacher_forced_span(
    tokenizer, prompt_prefix: str, response_text: str
) -> tuple[list[int], list[int]]:
    """Tokenize ``prompt_prefix + response_text`` and locate the response span.

    The two parts are concatenated *before* tokenization and aligned by char
    offsets, so a trailing space in the prefix merges into the response's
    leading-space token instead of becoming a lone ``" "`` token.
    """
    input_ids, offsets = encode_with_offsets(tokenizer, prompt_prefix + response_text)
    boundary = len(prompt_prefix)
    positions = [index for index, (start, end) in enumerate(offsets) if end > boundary]
    return input_ids, positions


def single_token_forms(tokenizer, word: str) -> dict[str, int]:
    """``{surface: token_id}`` over the single-token forms of ``word``
    (bare and leading-space); empty when neither form is a single token."""
    forms: dict[str, int] = {}
    for surface in (word, " " + word):
        token_ids = tokenizer.encode(surface, add_special_tokens=False)
        if len(token_ids) == 1:
            forms[surface] = token_ids[0]
    return forms


def preferred_single_token(tokenizer, word: str) -> tuple[int | None, str]:
    """``(token_id, surface)`` of ``word``'s single-token form, leading-space
    form preferred (the surface a word takes mid-sentence, where the
    swap/injection vectors act); ``(None, "")`` when no form is a single
    token."""
    forms = single_token_forms(tokenizer, word)
    for surface in (" " + word, word):
        if surface in forms:
            return forms[surface], surface
    return None, ""


def rank_and_prob(logits: torch.Tensor, token_ids: Sequence[int]) -> tuple[int, float]:
    """(min one-based rank, summed probability) of ``token_ids`` in a single
    next-token distribution ``logits`` of shape ``[vocab]``."""
    ids = [int(token_id) for token_id in token_ids]
    rank = min(one_based_rank(logits, token_id) for token_id in ids)
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    prob = float(log_probs[ids].exp().sum().item())
    return rank, prob


def word_variant_ids(tokenizer, word: str) -> list[int]:
    """Single-token ids over the case variants of ``word`` (as given,
    capitalized, lowercased), each in bare and leading-space form."""
    ids: list[int] = []
    for variant in (word, word.capitalize(), word.lower()):
        ids += list(single_token_forms(tokenizer, variant).values())
    return sorted(set(ids))


def number_target_ids(tokenizer, value) -> list[int]:
    """Single-token ids for an integer answer: digit string plus English
    number word (lowercase and capitalized), each in bare and leading-space
    form."""
    surfaces = [str(value)]
    try:
        word = number_word(int(value))
        surfaces += [word, word.capitalize()]
    except (TypeError, ValueError):
        pass
    ids: list[int] = []
    for surface in surfaces:
        ids += list(single_token_forms(tokenizer, surface).values())
    return sorted(set(ids))


def chat_prompt(carrier: str, instruction: str | None) -> tuple[str, str]:
    """Paper-style covert-task prompt, emulated as base-model plain text.

    Returns ``(prompt_prefix, response_text)`` with the carrier sentence
    teacher-forced as the assistant turn; ``instruction`` is the optional
    covert-task sentence (None for the no-instruction baseline).
    """
    middle = f" {instruction}" if instruction else ""
    prefix = (
        f'Human: Write "{carrier}"{middle} Don\'t write anything else.'
        f"\n\nAssistant:"
    )
    return prefix, " " + carrier


def load_experiment_data(slug: str) -> tuple[Path, dict, str]:
    """Load ``jacobian-lens/data/experiments/<slug>.json`` plus its sha256."""
    data_path = EXPERIMENT_DATA_DIR / f"{slug}.json"
    payload = json.loads(data_path.read_text(encoding="utf-8"))
    return data_path, payload, sha256_file(data_path)
