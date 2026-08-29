"""Load a fitted Jacobian lens, or a checkpoint together with its lens.

Fitted lenses live under ``fits/`` as
``fits/<model>/<model>_<revision>_jlens.pt``, with the Jacobians stored
float32 (storing them float16 is a lossy downcast of the fit; see
fit_lens.py). Lens files are mmapped rather than copied into RAM.

    from lens.load_lens import load_pair
    pair = load_pair("EleutherAI/pythia-6.9b", "step143000")
    logits = pair.lens_logits(residual, layer=20, method="jacobian")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

import jlens
from jlens.hf import Layout

from lens.load_models import REPO_ROOT, load_model

FITS_ROOT = REPO_ROOT / "fits"


def pythia_layout() -> Layout:
    """Pythia is GPT-NeoX; jlens needs the module names spelled out."""
    return Layout(
        path="gpt_neox",
        layers="layers",
        norm="final_layer_norm",
        embed="embed_in",
        lm_head="lm_head",
    )


def lens_path_for(model_id: str, revision: str) -> Path:
    name = model_id.replace("/", "_")
    return FITS_ROOT / name / f"{name}_{revision}_jlens.pt"


def load_lens(path: Path):
    """mmap a fitted lens instead of copying it into RAM."""
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found; download the released lens fits into fits/ "
            f"or fit them with lens/fit_lens.py"
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    return jlens.JacobianLens(
        jacobians=checkpoint["J"],
        n_prompts=checkpoint["n_prompts"],
        d_model=checkpoint["d_model"],
    )


@dataclass
class LoadedPair:
    """A checkpoint, its tokenizer, and the lens fitted to it."""

    model_id: str
    revision: str
    hf_model: object
    tokenizer: object
    lens: object
    lens_model: object
    device: torch.device
    _jacobians: dict = field(default_factory=dict, repr=False)

    @property
    def layers(self) -> list[int]:
        return list(self.lens.source_layers)

    def jacobian(self, layer: int) -> torch.Tensor:
        """J_l on the compute device, cached -- inner loops reuse it."""
        if layer not in self._jacobians:
            self._jacobians[layer] = self.lens.jacobians[layer].to(self.device, torch.float32)
        return self._jacobians[layer]

    def lens_logits(self, residual: torch.Tensor, layer: int, method: str = "jacobian"):
        """Lens readout for residuals ``[..., d_model]`` taken at ``layer``."""
        x = residual.to(self.device, torch.float32)
        if method == "jacobian":
            x = x @ self.jacobian(layer).T
        elif method != "logit":
            raise ValueError(f"method must be 'jacobian' or 'logit', got {method!r}")
        return self.lens_model.unembed(x)


def load_pair(model_id: str, revision: str, device: str | None = None) -> LoadedPair:
    """Load ``model_id@revision`` and its fitted lens, both float32."""
    lens = load_lens(lens_path_for(model_id, revision))
    hf_model, tokenizer = load_model(model_id, revision, device)
    lens_model = jlens.from_hf(hf_model, tokenizer, layout=pythia_layout(), compile=False)
    device_used = next(hf_model.parameters()).device
    return LoadedPair(model_id, revision, hf_model, tokenizer, lens, lens_model, device_used)
