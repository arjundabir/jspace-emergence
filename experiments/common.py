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


@dataclass
class LensRun:
    """Everything a per-lens experiment needs, loaded once per lens."""

    lens_path: Path
    model_id: str
    revision: str
    checkpoint_step: int
    lens: jlens.JacobianLens
    lens_model: object
    hf_model: object
    tokenizer: object
    device: torch.device
    all_layers: list[int] = field(default_factory=list)
    band: list[int] = field(default_factory=list)
    #: Device-resident float32 J_l cache: the same matrix sits in the
    #: innermost loop of every experiment, so it is moved to the compute
    #: device once per (lens, layer) instead of once per readout.
    _jacobian_cache: dict = field(default_factory=dict, repr=False)

    def transported(self, residual: torch.Tensor, layer: int, method: str):
        """``J_l @ h`` for the jacobian method, ``h`` unchanged for logit."""
        if method != "jacobian":
            return residual
        jacobian = self._jacobian_cache.get(layer)
        if jacobian is None or jacobian.device != residual.device:
            jacobian = self.lens.jacobians[layer].to(residual.device, torch.float32)
            self._jacobian_cache[layer] = jacobian
        return residual @ jacobian.T

    def lens_logits(self, residual: torch.Tensor, layer: int, method: str):
        """Lens readout logits for residuals ``[..., d_model]`` at ``layer``."""
        return unembed_stable(
            self.lens_model, self.transported(residual.float(), layer, method)
        )


def load_run(lens_path: Path, device: torch.device) -> LensRun:
    model_id, revision, checkpoint_step = identity_from_lens_filename(lens_path)
    lens = load_lens(lens_path)
    hf_model, tokenizer = load_model(model_id, revision, device)
    lens_model = jlens.from_hf(hf_model, tokenizer, layout=pythia_layout(), compile=False)
    all_layers = list(lens.source_layers)
    return LensRun(
        lens_path=lens_path,
        model_id=model_id,
        revision=revision,
        checkpoint_step=checkpoint_step,
        lens=lens,
        lens_model=lens_model,
        hf_model=hf_model,
        tokenizer=tokenizer,
        device=device,
        all_layers=all_layers,
        band=band_layers(all_layers, lens_model.n_layers),
    )


@torch.no_grad()
def capture_band_residuals(
    run: LensRun,
    input_ids: Sequence[int],
    positions: Sequence[int],
    layers: Sequence[int] | None = None,
) -> dict[int, torch.Tensor]:
    """One forward pass; residuals at ``positions`` for each requested layer.

    Returns ``{layer: [n_positions, d_model] float32}`` on the compute device.
    """
    layers = list(run.band if layers is None else layers)
    ids_tensor = torch.tensor(
        [list(input_ids)], dtype=torch.long, device=run.lens_model.input_device
    )
    with ActivationRecorder(run.lens_model.layers, at=layers) as recorder:
        run.lens_model.forward(ids_tensor)
        residuals = {
            layer: recorder.activations[layer][0, list(positions)].detach().float()
            for layer in layers
        }
    del ids_tensor
    return residuals


def lens_vector(
    lens, lens_model, layer: int, token_id: int, method: str, normalize: bool
) -> torch.Tensor:
    """Row ``token_id`` of ``W_U J_layer`` — the J-lens vector — on CPU float32.

    ``normalize`` is right for a projection (scale-invariant) and wrong for
    the swap, whose pseudo-inverse coordinates depend on the vectors' actual
    lengths.
    """
    unembed_row = lens_model._lm_head.weight[token_id].detach().float().cpu()
    if method == "jacobian":
        vector = lens.jacobians[layer].float().t() @ unembed_row
    else:
        vector = unembed_row.clone()
    if normalize:
        vector = vector / vector.norm().clamp_min(1e-8)
    return vector


def _split_output(output):
    if torch.is_tensor(output):
        return output, None
    return output[0], output[1:]


def _rebuild_output(hidden, rest):
    return hidden if rest is None else (hidden, *rest)


def make_swap_hook(
    v_source: torch.Tensor,
    v_target: torch.Tensor,
    positions: list[int],
    alpha: float,
):
    """h <- h + alpha * V (sigma(c) - c) with c = V^+ h and V = [v_s v_t].

    Reduces to h += alpha * (c_t - c_s) * (v_s - v_t); the component of h
    orthogonal to span{v_s, v_t} is left unchanged, as the paper specifies.
    """
    basis = torch.stack([v_source, v_target], dim=1)  # [d_model, 2]
    pseudo_inverse = torch.linalg.pinv(basis)  # [2, d_model]
    difference = v_source - v_target

    def hook(module, inputs, output):
        hidden, rest = _split_output(output)
        projector = pseudo_inverse.to(hidden.device, torch.float32)
        delta_direction = difference.to(hidden.device, torch.float32)
        hidden = hidden.clone()
        block = hidden[:, positions, :].float()
        coefficients = block @ projector.t()  # [batch, n_positions, 2]
        delta = (
            alpha
            * (coefficients[..., 1] - coefficients[..., 0]).unsqueeze(-1)
            * delta_direction
        )
        hidden[:, positions, :] = (block + delta).to(hidden.dtype)
        return _rebuild_output(hidden, rest)

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


def rebuild_summary(output_dir: Path, experiment_name: str) -> Path | None:
    """Rebuild ``emergence_summary_<experiment>.csv`` by scanning result JSONs."""
    rows: list[dict] = []
    for path in sorted(output_dir.resolve().rglob(f"*_{experiment_name}.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload["run_summary"]:
            rows.append({
                "experiment": experiment_name,
                "lens_filename": payload["lens_filename"],
                **row,
                "model_id": payload["model_id"],
                "revision": payload["revision"],
                "checkpoint_step": payload["checkpoint_step"],
                "parameter_count": payload["parameter_count"],
                "tokens_seen": payload["tokens_seen"],
                "band_first_layer": payload["band_first_layer"],
                "band_last_layer": payload["band_last_layer"],
                "lens_sha256": payload["lens_sha256"],
                "n_lens_fit_prompts": payload["n_lens_fit_prompts"],
            })
    if not rows:
        print(f"No {experiment_name} results found under {output_dir}")
        return None
    frame = pd.DataFrame(rows)
    sort_columns = [
        column for column in ("method", "parameter_count", "checkpoint_step")
        if column in frame.columns
    ]
    if sort_columns:
        frame = frame.sort_values(sort_columns, na_position="last")
    summary_path = output_dir / f"emergence_summary_{experiment_name}.csv"
    frame.to_csv(summary_path, index=False)
    print(f"{len(frame)} summary rows -> {summary_path}")
    return summary_path


def run_experiment(
    experiment_name: str,
    run_one: Callable[[LensRun], tuple[dict, dict[str, pd.DataFrame]]],
) -> int:
    """Shared main loop: discover lenses, run ``run_one`` per lens, write artifacts.

    ``run_one`` returns ``(payload, extra_csvs)``: ``payload`` must carry
    ``run_summary`` (a list of flat dicts, each with at least a ``method``
    key); ``extra_csvs`` maps a filename suffix (e.g. ``"_trial_scores.csv.gz"``)
    to a DataFrame. Provenance fields are filled in here.
    """
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lenses = discover_lenses(FITS_ROOT)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Experiment: {experiment_name}")
    print(f"Lenses: {len(lenses)}; output: {RESULTS_ROOT} (<model>/<step>/*_{experiment_name}.json)")

    for index, lens_path in enumerate(lenses, start=1):
        print(f"[{index}/{len(lenses)}] {lens_path.name}")
        started = time.perf_counter()
        run = load_run(lens_path, device)
        payload, extra_csvs = run_one(run)

        payload = {
            "experiment": experiment_name,
            "lens_filename": lens_path.name,
            "lens_sha256": sha256_file(lens_path),
            "model_id": run.model_id,
            "revision": run.revision,
            "checkpoint_step": run.checkpoint_step,
            "parameter_count": PARAM_COUNTS.get(run.model_id),
            "tokens_seen": run.checkpoint_step * TOKENS_PER_STEP,
            "fitted_layers": run.all_layers,
            "band_layers": run.band,
            "band_first_layer": run.band[0],
            "band_last_layer": run.band[-1],
            "n_model_layers": run.lens_model.n_layers,
            "n_lens_fit_prompts": int(run.lens.n_prompts),
            "methods": METHODS,
            "seed": RANDOM_SEED,
            "dtype": "float32",
            "device": device.type,
            **payload,
            "elapsed_seconds": round(time.perf_counter() - started, 2),
        }

        out_path = output_path_for_lens(lens_path, RESULTS_ROOT, experiment_name)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        for suffix, frame in extra_csvs.items():
            csv_path = output_path_for_lens(lens_path, RESULTS_ROOT, experiment_name, suffix=suffix)
            frame.to_csv(csv_path, index=False,
                         compression="gzip" if suffix.endswith(".gz") else None)
            payload.setdefault("csv_files", []).append(csv_path.name)
        out_path.write_text(
            json.dumps(sanitize_for_json(payload), indent=2, ensure_ascii=False,
                       default=json_default),
            encoding="utf-8",
        )
        print(
            f"  -> {out_path.name} ({payload['elapsed_seconds']}s)"
            + (f" {payload['headline']}" if payload.get("headline") else "")
        )

        del run
        if device.type == "cuda":
            torch.cuda.empty_cache()

    rebuild_summary(RESULTS_ROOT, experiment_name)
    return 0
