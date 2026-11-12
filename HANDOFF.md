# Handoff

Last updated: 2026-11-12, end of the first working session (1.7 h logged).

## Working conventions

- Felix learns and decides; the agent does all the execution (code, commands, commits,
  Notion updates) and explains each step.
- One roadmap step per turn, then stop and wait for Felix.
- Explanations to Felix are in Chinese. Everything inside the project (code, comments,
  commits, README, Notion) is in English.
- After each step:
  1. Add a Chinese section for the step to the walkthrough page (a claude.ai artifact; newest
     step first, progress rail at the top). Read it and republish to the same URL.
  2. Commit and push, then check that CI passes.
  3. Update the Notion roadmap: tick the checkbox, add Decision log entries with reasons,
     add an Hours log line and the new total.
- Explain from the concepts for a beginner: define every symbol, write formulas out step by
  step, and use real numbers from the project.

## Done

| Step | What | Commit |
| --- | --- | --- |
| Setup | uv project (Python 3.12), MIT license, ruff CI, public GitHub repo | `8ad09f1` |
| 1.1 | MiniMind cloned to `third_party/minimind` (commit `233d00fa`), `full_sft_768.pth` weights, inference checked on MPS | not in repo |
| 1.2 | Financial PhraseBank 75Agree, 5 duplicates dropped, stratified 70/15/15 split, seed 42 (2413 / 517 / 518) | `72647fc` |
| 1.3 | Metrics (accuracy, macro F1, ECE with 15 bins, NLL, Brier) and TF-IDF + logistic regression baseline | `6ee941d` |
| Model choice | Qwen3-0.6B is the main model (1.4); MiniMind is a one-run reference (1.5) | Notion |
| 1.4a-b | `peft` added; Qwen3-0.6B pinned at `c1899de289a04d12100db370d81485cdf75e47ca`; label-probability scoring; zero-shot check on validation | `9268316` |
| 1.4c | SFT examples: inference prompt + label word + `<\|im_end\|>`, loss on the answer only | `ccd3cb7` |

Key numbers:

- Baseline test: accuracy 0.851, macro F1 0.795, ECE 0.059. This is the bar to beat.
- Qwen3-0.6B zero-shot on validation: accuracy 0.288, macro F1 0.180, ECE 0.569. It answers
  "positive" for 501 of 517 sentences and is very sensitive to prompt wording.

## Code map

- `src/finsent_calib/data_prep.py`: download, dedup, split. Writes `data/processed/*.csv`.
- `src/finsent_calib/metrics.py`: `evaluate(probs, y_true)` for every model.
- `src/finsent_calib/baseline.py`: baseline grid search on validation, one test evaluation.
- `src/finsent_calib/llm.py`: `load_qwen`, `build_prompt`, `label_token_ids`, `label_probs`
  (left padding with explicit position ids; softmax over the three label tokens).
- `src/finsent_calib/qwen_zeroshot.py`: zero-shot run on validation only.
- `src/finsent_calib/sft.py`: `build_example` for training.
- `results/`: committed metrics and predictions (sentence ids and probabilities only).

## Environment notes

- Run `uv run python -m finsent_calib.data_prep` once on a fresh clone to rebuild
  `data/processed/` (gitignored).
- Qwen weights live in the global Hugging Face cache, not in the repo.
- The saved Hugging Face token is read-only. Uploading a model later needs a fine-grained
  write token, which Felix creates and logs in with; ask when that step comes.
- CI runs ruff only (no tests, so torch is not installed in CI). It prints a Node 20
  deprecation warning for `actions/checkout@v4` and `astral-sh/setup-uv@v6`.
- The test set has been used exactly once (baseline). Do not use it for any tuning.

## Next step: 1.4d, add LoRA to Qwen3-0.6B with PEFT

- Build a `LoraConfig` and wrap the model with `get_peft_model`. Starting point to discuss:
  rank 16, alpha 32, dropout 0.05, targets `q_proj`, `k_proj`, `v_proj`, `o_proj`,
  `gate_proj`, `up_proj`, `down_proj`.
- Print the trainable parameter count and share, and compare with MiniMind's own LoRA
  (only square `q_proj` / `o_proj`, no alpha scaling, 393,216 parameters).
- Then 1.4e: a short training run on the Mac with a few hundred training examples
  (max length 256, MPS, float32), checking that the loss goes down and validation metrics
  move. Then 1.4f: the full run on Colab, hyperparameters chosen on validation, one test
  evaluation.
