from ablations.common import Task, run, target_boundary_prepare

TASK = Task(
    name="order_ops",
    slug="lens-eval-order-ops",
    prepare_example=target_boundary_prepare,
)

if __name__ == "__main__":
    raise SystemExit(run(TASK))
