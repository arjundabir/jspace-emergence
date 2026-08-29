"""Multilingual evaluation: readout at the token immediately preceding
``target``, with up to four intermediates per item (language, relation,
English in/out).

Several intermediates are endonyms (espanol, portugues) that are multi-token
under the GPT-NeoX BPE; those are reported in tokenization_coverage and
excluded from the denominator rather than scored as failures.

    python -m evals.multilingual
"""

from evals.common import Task, run, target_boundary_prepare

TASK = Task(
    name="multilingual",
    slug="lens-eval-multilingual",
    readout_mode="target_boundary",
    prepare_item=target_boundary_prepare,
)

if __name__ == "__main__":
    raise SystemExit(run(TASK))
