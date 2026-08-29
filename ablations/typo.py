from ablations.common import Task, prompt_final_prepare, run

TASK = Task(
    name="typo",
    slug="lens-eval-typo",
    prepare_example=prompt_final_prepare,
)

if __name__ == "__main__":
    raise SystemExit(run(TASK))
