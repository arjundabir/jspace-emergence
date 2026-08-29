"""Typo ablation.

Encode the prompt only; readout is the last prompt token (the last BPE
fragment of the misspelling). Usable when the correction
(``intermediates[0]``) is single-token; the correction itself is scored as
the answer, and its direction is ablated across the workspace band.

    python -m ablations.typo --model EleutherAI/pythia-70m
"""

from ablations.common import Task, prompt_final_prepare, run

TASK = Task(
    name="typo",
    slug="lens-eval-typo",
    prepare_example=prompt_final_prepare,
)

if __name__ == "__main__":
    raise SystemExit(run(TASK))
