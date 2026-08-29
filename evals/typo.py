from evals.common import Task, prompt_final_prepare, run

TASK = Task(
    name="typo",
    slug="lens-eval-typo",
    readout_mode="prompt_final",
    prepare_item=prompt_final_prepare,
)

if __name__ == "__main__":
    raise SystemExit(run(TASK))
