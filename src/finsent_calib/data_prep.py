"""
Download Financial PhraseBank and Build a fixed, stratified train/val/test split.
"""

import hashlib
import json
import zipfile
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split

REPO_ID = "takala/financial_phrasebank"
REVISION = "8d3fe0c36d5feec6b3cc5e455b0fcb4820fb9964"
ZIP_IN_REPO = "data/FinancialPhraseBank-v1.0.zip"
SUBSET_FILE = "FinancialPhraseBank-v1.0/Sentences_75Agree.txt"
LABELS = ["negative", "neutral", "positive"]
SEED = 42

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"


def download_raw() -> Path:
    """Download the official zip at a pinned revision and return its local path."""
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=ZIP_IN_REPO,
        repo_type="dataset",
        revision=REVISION,
        local_dir=RAW_DIR,
    )
    return Path(path)


def load_sentences(zip_path: Path) -> tuple[pd.DataFrame, str]:
    """Read the 75Agree file, drop exact duplicates, and return the table and file sha256."""
    with zipfile.ZipFile(zip_path) as zf:
        raw_bytes = zf.read(SUBSET_FILE)
    sha256 = hashlib.sha256(raw_bytes).hexdigest()

    rows = []
    for line_no, line in enumerate(raw_bytes.decode("latin-1").splitlines()):
        if not line.strip():
            continue
        sentence, label = line.rsplit("@", 1)
        rows.append({"id": line_no, "sentence": sentence.strip(), "label": label.strip()})
    df = pd.DataFrame(rows)

    assert set(df["label"]) == set(LABELS), f"unexpected labels: {set(df['label'])}"
    n_before = len(df)
    df = df.drop_duplicates(subset="sentence", keep="first").reset_index(drop=True)
    print(f"read {n_before} rows, dropped {n_before - len(df)} duplicates, kept {len(df)}")
    return df, sha256


def make_splits(df: pd.DataFrame, seed: int = SEED) -> dict[str, pd.DataFrame]:
    """Stratified 70/15/15 split: first 70/30, then the 30% halved into val and test."""
    train, rest = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=seed)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest["label"], random_state=seed)
    return {"train": train, "val": val, "test": test}


def class_balance(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Count and share of each label in each split."""
    rows = []
    for name, part in splits.items():
        counts = part["label"].value_counts()
        for label in LABELS:
            n = int(counts.get(label, 0))
            rows.append({"split": name, "label": label, "count": n, "share": n / len(part)})
    return pd.DataFrame(rows)


def main() -> None:
    zip_path = download_raw()
    df, sha256 = load_sentences(zip_path)
    splits = make_splits(df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.sort_values("id").to_csv(PROCESSED_DIR / f"{name}.csv", index=False)

    balance = class_balance(splits)
    balance.to_csv(RESULTS_DIR / "data_split_balance.csv", index=False, float_format="%.4f")

    manifest = {
        "source": f"https://huggingface.co/datasets/{REPO_ID}",
        "revision": REVISION,
        "subset_file": SUBSET_FILE,
        "subset_sha256": sha256,
        "seed": SEED,
        "n_after_dedup": len(df),
        "sizes": {name: len(part) for name, part in splits.items()},
    }
    (RESULTS_DIR / "data_split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(json.dumps(manifest["sizes"]))
    print(balance.pivot(index="split", columns="label", values="share").round(3))


if __name__ == "__main__":
    main()
