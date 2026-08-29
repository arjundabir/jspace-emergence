"""Multilingual ablation.

Same ablation and KL method as multihop, with whole-answer scoring that
keeps multi-token final targets usable: the full target span is scored by
teacher-forced length-normalized mean log-probability, and rank is reported
only for single-token targets. Usable when the first intermediate is
single-token.

    python -m ablations.multilingual
"""

from ablations.common import Task, run, whole_answer_prepare

TASK = Task(
    name="multilingual",
    slug="lens-eval-multilingual",
    prepare_example=whole_answer_prepare,
    whole_answer=True,
)

if __name__ == "__main__":
    raise SystemExit(run(TASK))
