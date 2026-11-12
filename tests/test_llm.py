import pytest
from transformers import AutoTokenizer

from finsent_calib.llm import QWEN_ID, QWEN_REVISION, build_prompt, label_token_ids


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(QWEN_ID, revision=QWEN_REVISION)


def test_label_words_are_distinct_single_tokens(tokenizer) -> None:
    ids = label_token_ids(tokenizer)
    assert len(ids) == 3
    assert len(set(ids)) == 3


def test_prompt_ends_where_the_answer_starts(tokenizer) -> None:
    prompt = build_prompt(tokenizer, "Sales rose 5 %.")
    assert "Sentence: Sales rose 5 %." in prompt
    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
