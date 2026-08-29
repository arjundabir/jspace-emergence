from __future__ import annotations

import argparse
import gc
import logging
import sys
import time

import torch

from lens.load_lens import lens_path_for, pythia_layout
from lens.load_models import CHECKPOINT_STEPS, load_model

logger = logging.getLogger("fit_lens")

DEFAULT_MODEL = "EleutherAI/pythia-70m"
N_PROMPTS = 1000
MAX_SEQ_LEN = 128


def fit_one(model_id: str, step: int, prompts, dim_batch: int, device: torch.device) -> None:
    import jlens

    lens_path = lens_path_for(model_id, f"step{step}")
    lens_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    hf_model, tokenizer = load_model(model_id, f"step{step}", device)
    lens_model = jlens.from_hf(hf_model, tokenizer, layout=pythia_layout(), compile=False)
    lens = jlens.fit(
        lens_model,
        prompts=prompts,
        max_seq_len=MAX_SEQ_LEN,
        dim_batch=dim_batch,
    )
    # JacobianLens.save defaults to float16, a lossy downcast of the fit; pin fp32.
    lens.save(str(lens_path), dtype=torch.float32)
    logger.info("step%-7d fitted in %.1fs -> %s", step, time.perf_counter() - t0, lens_path.name)

    del hf_model, lens_model, lens
    gc.collect()
    torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dim-batch", type=int, default=512,
                        help="Jacobian columns per backward pass; lower it to fit "
                             "GPU memory on the larger models")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # fp32 end to end: TF32 would cut matmul inputs to 10 mantissa bits.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from jlens.examples import load_wikitext_prompts

    logger.info("loading %d wikitext prompts", N_PROMPTS)
    prompts = load_wikitext_prompts(N_PROMPTS, min_chars=600)

    logger.info("fitting %s at fp32: %d checkpoints, dim_batch=%d",
                args.model, len(CHECKPOINT_STEPS), args.dim_batch)
    for step in CHECKPOINT_STEPS:
        fit_one(args.model, step, prompts, args.dim_batch, device)
    logger.info("done: %d checkpoints", len(CHECKPOINT_STEPS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
