"""Multi-hop evaluation: readout at the token immediately preceding ``target``.

The bridging entity of a two-hop factual chain is scored at the boundary
position, where the model is about to emit the target; the intermediate
appearing in the lens there means the hop was resolved internally. The target
itself is the capability control's answer.

    python -m evals.multihop
    python -m evals.multihop --limit 1 --force
"""

from evals.common import Task, run, target_boundary_prepare

TASK = Task(
    name="multihop",
    slug="lens-eval-multihop",
    readout_mode="target_boundary",
    prepare_item=target_boundary_prepare,
)

if __name__ == "__main__":
    raise SystemExit(run(TASK))
