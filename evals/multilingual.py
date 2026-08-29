from evals.common import Task, run, target_boundary_prepare

TASK = Task(
    name="multilingual",
    slug="lens-eval-multilingual",
    readout_mode="target_boundary",
    prepare_item=target_boundary_prepare,
)

if __name__ == "__main__":
    raise SystemExit(run(TASK))
