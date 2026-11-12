import pytest
from transformers import AutoTokenizer

from finsent_calib.llm import QWEN_ID, QWEN_REVISION, build_prompt, label_token_ids
from finsent_calib.sft import IGNORE_INDEX, build_example


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(QWEN_ID, revision=QWEN_REVISION)


def test_loss_is_only_on_the_answer(tokenizer) -> None:
    example = build_example(tokenizer, "Sales rose 5 %.", "positive")
    trained = [t for t in example["labels"] if t != IGNORE_INDEX]
    assert tokenizer.decode(trained) == "positive<|im_end|>"


def test_prompt_part_matches_the_inference_prompt(tokenizer) -> None:
    example = build_example(tokenizer, "Sales rose 5 %.", "negative")
    n_prompt = example["labels"].count(IGNORE_INDEX)
    assert tokenizer.decode(example["input_ids"][:n_prompt]) == build_prompt(
        tokenizer, "Sales rose 5 %."
    )


def test_first_answer_token_is_the_scored_label_token(tokenizer) -> None:
    for label, label_id in zip(
        ["negative", "neutral", "positive"], label_token_ids(tokenizer), strict=True
    ):
        example = build_example(tokenizer, "Sales rose 5 %.", label)
        n_prompt = example["labels"].count(IGNORE_INDEX)
        assert example["input_ids"][n_prompt] == label_id


def test_lengths_line_up(tokenizer) -> None:
    example = build_example(tokenizer, "Sales rose 5 %.", "neutral")
    assert len(example["input_ids"]) == len(example["attention_mask"]) == len(example["labels"])
