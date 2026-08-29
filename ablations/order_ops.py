"""Order-of-operations ablation.

Same protocol as multihop: encode ``prompt + target``, readout at the token
before the target, ablate the literal ``intermediates[0]`` direction (no
synonym expansion), and score the target; both must be single-token.

    python -m ablations.order_ops --model EleutherAI/pythia-70m
"""

from ablations.common import Task, run, target_boundary_prepare

TASK = Task(
    name="order_ops",
    slug="lens-eval-order-ops",
    prepare_example=target_boundary_prepare,
)

if __name__ == "__main__":
    raise SystemExit(run(TASK))
