"""Poetry ablation.

Poetry items have no separate ``target``; the rhyme word
(``intermediates[0]``) is both the ablated direction and the answer. KL is
taken at the readout before the answer span, and the whole answer is scored
by teacher-forced length-normalized mean log-probability, so multi-token
rhyme words stay usable; rank is additionally reported for single-token ones.

    python -m ablations.poetry
"""

from ablations.common import Task, run, whole_answer_prepare

TASK = Task(
    name="poetry",
    slug="lens-eval-poetry",
    prepare_example=whole_answer_prepare,
    whole_answer=True,
)

if __name__ == "__main__":
    raise SystemExit(run(TASK))
