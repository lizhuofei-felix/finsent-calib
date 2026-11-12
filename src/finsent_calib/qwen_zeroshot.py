"""Zero-shot Qwen3-0.6B on the validation set, before any fine-tuning. The test set is not used."""

import json

from finsent_calib.baseline import PRED_DIR, load_split, save_predictions
from finsent_calib.data_prep import RESULTS_DIR
from finsent_calib.llm import INSTRUCTION, QWEN_ID, QWEN_REVISION, label_probs, load_qwen
from finsent_calib.metrics import evaluate

OUT_DIR = RESULTS_DIR / "qwen_zeroshot"


def main() -> None:
    val = load_split("val")
    model, tokenizer = load_qwen()
    probs, coverage = label_probs(model, tokenizer, val["sentence"].tolist())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    save_predictions(val, probs, PRED_DIR / "qwen_zeroshot_val.csv")
    report = {
        "model": QWEN_ID,
        "revision": QWEN_REVISION,
        "instruction": INSTRUCTION,
        "coverage_mean": float(coverage.mean()),
        "coverage_min": float(coverage.min()),
        "val": evaluate(probs, val["y"].to_numpy()),
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"coverage mean {report['coverage_mean']:.3f}, min {report['coverage_min']:.3f}")
    for name, value in report["val"].items():
        print(f"val {name}: {value if name == 'confusion_matrix' else round(value, 4)}")


if __name__ == "__main__":
    main()
