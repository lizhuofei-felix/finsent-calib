"""Supervised fine-tuning (SFT) examples: a chat prompt plus a one-word answer.

The loss is computed on the answer only.
"""

from transformers import PreTrainedTokenizerBase

from finsent_calib.llm import build_prompt

IGNORE_INDEX = -100  # label value that PyTorch's cross-entropy loss skips


def build_example(tokenizer: PreTrainedTokenizerBase, sentence: str, label: str) -> dict:
    """Token ids for one training example.

    input_ids = prompt tokens + answer tokens, where the answer is the label word followed by
    the end-of-turn token. labels copies input_ids but sets every prompt position to
    IGNORE_INDEX, so the loss is computed only on the answer.
    """
    prompt_ids = tokenizer(build_prompt(tokenizer, sentence), add_special_tokens=False).input_ids
    answer_ids = tokenizer(label + "<|im_end|>", add_special_tokens=False).input_ids
    return {
        "input_ids": prompt_ids + answer_ids,
        "attention_mask": [1] * (len(prompt_ids) + len(answer_ids)),
        "labels": [IGNORE_INDEX] * len(prompt_ids) + answer_ids,
    }
