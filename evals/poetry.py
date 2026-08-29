from evals.common import Task, encode_with_offsets, run


def last_newline_prepare(item_index: int, item: dict, tokenizer) -> dict:
    prompt = item["prompt"]
    rhyme_word = item["intermediates"][0]
    evaluation_text = prompt + rhyme_word
    input_ids, offsets = encode_with_offsets(tokenizer, evaluation_text)
    boundary = len(prompt)
    answer_first = next(
        index for index, (start, end) in enumerate(offsets) if end > boundary
    )
    newline_char = prompt.rfind("\n")
    readout_position = next(
        index for index, (start, end) in enumerate(offsets)
        if start <= newline_char < end
    )
    return {
        "item_index": item_index,
        "name": item["name"],
        "input_ids": input_ids,
        "readout_position": readout_position,
        "answer_position": answer_first - 1,
        "answer_word": rhyme_word,
        # Ablating the rhyme direction at the position that emits it would
        # trivially move the output, so ablations exclude the last position.
        "exclude_last_from_ablation": True,
        "intermediates": item["intermediates"],
    }


TASK = Task(
    name="poetry",
    slug="lens-eval-poetry",
    readout_mode="last_newline",
    prepare_item=last_newline_prepare,
)

if __name__ == "__main__":
    raise SystemExit(run(TASK))
