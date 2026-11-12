# finsent-calib

**Calibrated financial sentiment signal with a LoRA fine-tuned small LLM.**

> Status: work in progress. Results tables will be filled in as each stage is completed.

## Overview

This project fine-tunes a small language model to classify the sentiment of financial news,
checks whether the model's confidence is well calibrated, and tests whether the sentiment
score works as a trading signal. Finance is the task; statistics is the method.

## Research questions

1. Does a LoRA fine-tuned small LLM beat a TF-IDF + logistic regression baseline on
   financial sentiment classification?
2. How do accuracy and calibration change with the size of the fine-tuning set, measured
   over several seeds with bootstrap confidence intervals?
3. Does the model's sentiment score predict next-day stock returns, measured by
   cross-sectional rank IC with Newey-West standard errors?

## Method

| Stage | Content |
| --- | --- |
| 1. Fine-tune | TF-IDF + logistic regression baseline; LoRA fine-tune of [MiniMind](https://github.com/jingyaogong/minimind). Predictions read the next-token logits of the three label words, so every prediction has a proper probability. |
| 2. Statistics | Data-size experiment (250 to full, 3 seeds each); bootstrap CIs; paired bootstrap and McNemar's test against the baseline; reliability diagrams and temperature scaling. |
| 3. Quant test | Daily sentiment factor per stock, next-day Spearman rank IC, ICIR, Newey-West t-statistic, compared with the baseline factor and a shuffled placebo. |

Metrics: accuracy, macro F1, expected calibration error (ECE, 15 equal-width bins),
negative log-likelihood (NLL) and Brier score.

## Results

To be added.

## Repository layout

```text
configs/     one config file per experiment run
notebooks/   Colab notebooks for training and experiments
results/     metric CSVs and figures (committed)
src/finsent_calib/   data preparation, baseline, training, prediction, calibration, factor IC
tests/       unit tests
data/        raw and processed data (not committed)
```

## Setup

The environment is managed with [uv](https://docs.astral.sh/uv/). Python 3.12.

```bash
uv sync
uv run pytest
```

Heavy training runs on Google Colab; setup, debugging and quick checks run locally
(Apple Silicon, PyTorch MPS).

## Data and licenses

- Code in this repository: MIT License.
- [Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank)
  (Malo et al., 2014): CC BY-NC-SA 3.0, non-commercial. The data is downloaded at run time
  and is not redistributed here.
- [MiniMind](https://github.com/jingyaogong/minimind): Apache-2.0. Used as an external
  dependency, not vendored.

## Limitations

To be completed with the results. Known in advance: small stock universe, survivorship bias,
and no transaction costs in the first version of the factor test.
