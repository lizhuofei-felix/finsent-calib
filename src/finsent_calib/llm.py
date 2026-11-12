"""Prompt building and label-probability scoring for the Qwen3 causal language model."""

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from finsent_calib.data_prep import LABELS

QWEN_ID = "Qwen/Qwen3-0.6B"
QWEN_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
INSTRUCTION = (
    "Classify the sentiment of this financial news sentence from an investor's point of view. "
    "Answer with one word: positive, negative or neutral.\n\nSentence: "
)


def pick_device() -> str:
    """Use a CUDA GPU (Colab) if present, else Apple's MPS GPU, else the CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_qwen(device: str | None = None) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load Qwen3-0.6B and its tokenizer at the pinned revision."""
    tokenizer = AutoTokenizer.from_pretrained(QWEN_ID, revision=QWEN_REVISION)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_ID, revision=QWEN_REVISION, dtype=torch.float32
    )
    return model.to(device or pick_device()).eval(), tokenizer


def build_prompt(tokenizer: PreTrainedTokenizerBase, sentence: str) -> str:
    """Chat-formatted prompt that ends right where the assistant's one-word answer starts."""
    messages = [{"role": "user", "content": INSTRUCTION + sentence}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def label_token_ids(tokenizer: PreTrainedTokenizerBase) -> list[int]:
    """Token id of each label word, in LABELS order; each word must be exactly one token."""
    ids = [tokenizer.encode(label, add_special_tokens=False) for label in LABELS]
    assert all(len(i) == 1 for i in ids), f"label words must be single tokens: {ids}"
    return [i[0] for i in ids]


@torch.no_grad()
def label_probs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    sentences: list[str],
    batch_size: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Probabilities of the three labels and the coverage of each sentence.

    probs: N x 3, softmax over the three label-word logits at the answer position.
    coverage: N, the share of the full next-token distribution that falls on the three words.
    """
    ids = label_token_ids(tokenizer)
    probs, coverage = [], []
    for start in range(0, len(sentences), batch_size):
        prompts = [build_prompt(tokenizer, s) for s in sentences[start : start + batch_size]]
        enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        position_ids = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
        logits = model(**enc, position_ids=position_ids).logits[:, -1, :].float()
        coverage.append(torch.softmax(logits, dim=-1)[:, ids].sum(-1).cpu())
        probs.append(torch.softmax(logits[:, ids], dim=-1).cpu())
    return torch.cat(probs).numpy(), torch.cat(coverage).numpy()
