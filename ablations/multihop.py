from ablations.common import Task, run, target_boundary_prepare

TASK = Task(
    name="multihop",
    slug="lens-eval-multihop",
    prepare_example=target_boundary_prepare,
)

if __name__ == "__main__":
    raise SystemExit(run(TASK))
