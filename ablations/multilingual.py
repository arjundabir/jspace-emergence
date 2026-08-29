from ablations.common import Task, run, whole_answer_prepare

TASK = Task(
    name="multilingual",
    slug="lens-eval-multilingual",
    prepare_example=whole_answer_prepare,
    whole_answer=True,
)

if __name__ == "__main__":
    raise SystemExit(run(TASK))
