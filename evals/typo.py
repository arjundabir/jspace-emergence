"""Typo evaluation: readout at the final prompt token, the last BPE fragment
of the misspelled word.

The model's actual next token there is whatever follows the typo, so the
correct spelling appearing in the lens is evidence of internal correction
rather than of imminent output. There is no answer target.

    python -m evals.typo
"""

from evals.common import Task, prompt_final_prepare, run

TASK = Task(
    name="typo",
    slug="lens-eval-typo",
    readout_mode="prompt_final",
    prepare_item=prompt_final_prepare,
)

if __name__ == "__main__":
    raise SystemExit(run(TASK))
