"""Association ablation.

Encode the prompt only; readout is the last prompt token. Usable when the
first intermediate (the latent concept) is single-token; the concept itself is
scored as the answer. Its normalized J-lens direction is ablated across the
workspace band at all prompt positions.

    python -m ablations.association --model EleutherAI/pythia-70m
"""

from ablations.common import Task, prompt_final_prepare, run

TASK = Task(
    name="association",
    slug="lens-eval-association",
    prepare_example=prompt_final_prepare,
)

if __name__ == "__main__":
    raise SystemExit(run(TASK))
