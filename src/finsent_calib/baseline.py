"""TF-IDF + logistic regression baseline, tuned on the validation set only."""

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline

from finsent_calib.data_prep import LABELS, PROCESSED_DIR, RESULTS_DIR, SEED
from finsent_calib.metrics import evaluate

C_GRID = [0.1, 1.0, 10.0, 100.0, 1000.0]
NGRAM_MAX_GRID = [1, 2]
OUT_DIR = RESULTS_DIR / "baseline"
PRED_DIR = RESULTS_DIR / "predictions"


def load_split(name: str) -> pd.DataFrame:
    """Read one processed split and add the integer label column y."""
    df = pd.read_csv(PROCESSED_DIR / f"{name}.csv")
    df["y"] = df["label"].map({label: i for i, label in enumerate(LABELS)})
    return df


def build_model(c: float, ngram_max: int) -> Pipeline:
    """TF-IDF features followed by multinomial logistic regression."""
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, ngram_max), sublinear_tf=True),
        LogisticRegression(C=c, max_iter=5000, random_state=SEED),
    )


def save_predictions(df: pd.DataFrame, probs: np.ndarray, path: Path) -> None:
    """Save sentence ids and the three class probabilities (no text, no labels)."""
    out = pd.DataFrame(probs, columns=[f"p_{label}" for label in LABELS])
    out.insert(0, "id", df["id"].to_numpy())
    out.to_csv(path, index=False, float_format="%.6f")


def main() -> None:
    train, val, test = (load_split(name) for name in ("train", "val", "test"))

    # 1. Grid search: fit on train, score on validation. The test set is not touched here.
    rows = []
    for c, ngram_max in itertools.product(C_GRID, NGRAM_MAX_GRID):
        model = build_model(c, ngram_max).fit(train["sentence"], train["y"])
        metrics = evaluate(model.predict_proba(val["sentence"]), val["y"].to_numpy())
        rows.append({"C": c, "ngram_max": ngram_max, **metrics})
    search = pd.DataFrame(rows).drop(columns="confusion_matrix")
    search = search.sort_values(["macro_f1", "nll"], ascending=[False, True])
    best = search.iloc[0]
    print(search.round(4).to_string(index=False))

    # 2. Refit the chosen setting on train, then score validation and test once.
    model = build_model(float(best["C"]), int(best["ngram_max"]))
    model.fit(train["sentence"], train["y"])
    assert list(model.classes_) == [0, 1, 2]
    val_probs = model.predict_proba(val["sentence"])
    test_probs = model.predict_proba(test["sentence"])

    # 3. Save the search table, the predicted probabilities and the final metrics.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    search.to_csv(OUT_DIR / "val_search.csv", index=False, float_format="%.4f")
    save_predictions(val, val_probs, PRED_DIR / "baseline_val.csv")
    save_predictions(test, test_probs, PRED_DIR / "baseline_test.csv")

    report = {
        "model": "tfidf_logreg",
        "chosen": {"C": float(best["C"]), "ngram_max": int(best["ngram_max"])},
        "selection": "highest validation macro F1, ties broken by lower validation NLL",
        "val": evaluate(val_probs, val["y"].to_numpy()),
        "test": evaluate(test_probs, test["y"].to_numpy()),
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["chosen"]))
    for name, value in report["test"].items():
        print(f"test {name}: {value if name == 'confusion_matrix' else round(value, 4)}")


if __name__ == "__main__":
    main()
