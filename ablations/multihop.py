"""Multi-hop ablation.

Encode ``prompt + target``; readout is the token immediately preceding the
target. The bridging entity (``intermediates[0]``) supplies the ablated
direction and the target is scored; both must be single-token.

    python -m ablations.multihop --model EleutherAI/pythia-70m
"""

from ablations.common import Task, run, target_boundary_prepare

TASK = Task(
    name="multihop",
    slug="lens-eval-multihop",
    prepare_example=target_boundary_prepare,
)

if __name__ == "__main__":
    raise SystemExit(run(TASK))
