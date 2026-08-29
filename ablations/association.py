from ablations.common import Task, prompt_final_prepare, run

TASK = Task(
    name="association",
    slug="lens-eval-association",
    prepare_example=prompt_final_prepare,
)

if __name__ == "__main__":
    raise SystemExit(run(TASK))
