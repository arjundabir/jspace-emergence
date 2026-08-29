"""Association evaluation: readout at the final prompt token (the closing period).

The vignette never names the concept and the model is not about to say it, so
anything the lens surfaces at the final token was assembled from accumulated
context. There is no answer target; the capability control records only the
model's own top-1.

    python -m evals.association
"""

from evals.common import Task, prompt_final_prepare, run

TASK = Task(
    name="association",
    slug="lens-eval-association",
    readout_mode="prompt_final",
    prepare_item=prompt_final_prepare,
)

if __name__ == "__main__":
    raise SystemExit(run(TASK))
