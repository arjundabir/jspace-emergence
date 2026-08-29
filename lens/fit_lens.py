"""Fit Jacobian lenses at float32, one checkpoint at a time, on a single GPU.

Everything is float32 end to end: TF32 matmuls are disabled, checkpoints load
as fp32 (exact at every size; see load_models.py), and the fitted Jacobians
are stored fp32 (``JacobianLens.save`` defaults to float16, which is a lossy
downcast of the fit).

    python -m lens.fit_lens                                  # 70m, all steps
    python -m lens.fit_lens --model EleutherAI/pythia-410m
    python -m lens.fit_lens --steps 100000,130000,143000 --force
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
import time

import torch

from lens.load_lens import lens_path_for, pythia_layout
from lens.load_models import CHECKPOINT_STEPS, REPO_ROOT, load_model

logger = logging.getLogger("fit_lens")

DEFAULT_MODEL = "EleutherAI/pythia-70m"
FIT_CHECKPOINT_DIR = REPO_ROOT / ".cache" / "fit_checkpoints"


def fit_one(model_id: str, step: int, prompts, args, device: torch.device) -> None:
    import jlens

    lens_path = lens_path_for(model_id, f"step{step}")
    if lens_path.is_file() and not args.force:
        logger.info("skip step%d (lens exists)", step)
        return
    name = model_id.replace("/", "_")
    fit_ckpt = FIT_CHECKPOINT_DIR / name / f"{name}_step{step}_fit.ckpt"
    fit_ckpt.parent.mkdir(parents=True, exist_ok=True)
    lens_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    hf_model, tokenizer = load_model(model_id, f"step{step}", device)
    lens_model = jlens.from_hf(hf_model, tokenizer, layout=pythia_layout(), compile=False)
    lens = jlens.fit(
        lens_model,
        prompts=prompts,
        max_seq_len=args.max_seq_len,
        dim_batch=args.dim_batch,
        checkpoint_path=str(fit_ckpt),
        checkpoint_every=args.checkpoint_every,
        resume=not args.force,
    )
    # JacobianLens.save defaults to float16, a lossy downcast of the fit; pin fp32.
    lens.save(str(lens_path), dtype=torch.float32)
    logger.info("step%-7d fitted in %.1fs -> %s", step, time.perf_counter() - t0, lens_path.name)

    del hf_model, lens_model, lens
    gc.collect()
    torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--steps", default=None,
                   help=f"Comma-separated steps; default all {len(CHECKPOINT_STEPS)}")
    p.add_argument("--n-prompts", type=int, default=1000)
    p.add_argument("--max-seq-len", type=int, default=128)
    p.add_argument("--dim-batch", type=int, default=512)
    p.add_argument("--checkpoint-every", type=int, default=100)
    p.add_argument("--device", default="cuda")
    p.add_argument("--force", action="store_true",
                   help="Re-fit even if the lens exists; ignores fit checkpoints")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # fp32 end to end: TF32 would cut matmul inputs to 10 mantissa bits.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    device = torch.device(args.device)
    steps = ([int(s) for s in args.steps.split(",") if s.strip()]
             if args.steps else list(CHECKPOINT_STEPS))

    from jlens.examples import load_wikitext_prompts

    logger.info("loading %d wikitext prompts", args.n_prompts)
    prompts = load_wikitext_prompts(args.n_prompts, min_chars=600)
    if len(prompts) < args.n_prompts:
        logger.warning("got %d prompts, wanted %d", len(prompts), args.n_prompts)

    logger.info("fitting %s at fp32: %d checkpoints, dim_batch=%d, seq_len=%d",
                args.model, len(steps), args.dim_batch, args.max_seq_len)
    for step in steps:
        fit_one(args.model, step, prompts, args, device)
    logger.info("done: %d checkpoints", len(steps))
    return 0


if __name__ == "__main__":
    sys.exit(main())
