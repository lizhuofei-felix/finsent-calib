import pandas as pd

from finsent_calib.data_prep import class_balance, make_splits


def make_toy_data() -> pd.DataFrame:
    labels = ["neutral"] * 60 + ["positive"] * 28 + ["negative"] * 12
    return pd.DataFrame(
        {"id": range(100), "sentence": [f"sentence {i}" for i in range(100)], "label": labels}
    )


def test_splits_are_disjoint_and_complete() -> None:
    df = make_toy_data()
    splits = make_splits(df, seed=0)
    ids = [set(part["id"]) for part in splits.values()]
    assert ids[0].isdisjoint(ids[1])
    assert ids[0].isdisjoint(ids[2])
    assert ids[1].isdisjoint(ids[2])
    assert set().union(*ids) == set(df["id"])


def test_split_sizes_are_70_15_15() -> None:
    splits = make_splits(make_toy_data(), seed=0)
    sizes = {name: len(part) for name, part in splits.items()}
    assert sizes == {"train": 70, "val": 15, "test": 15}


def test_split_is_stratified() -> None:
    balance = class_balance(make_splits(make_toy_data(), seed=0))
    shares = balance.pivot(index="split", columns="label", values="share")
    assert (shares.max() - shares.min()).max() < 0.05


def test_same_seed_gives_same_split() -> None:
    df = make_toy_data()
    first = make_splits(df, seed=0)["test"]["id"].tolist()
    second = make_splits(df, seed=0)["test"]["id"].tolist()
    assert first == second
