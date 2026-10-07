"""Stage 1: load the dataset, split it, write CSVs (tracked by DVC)."""
from pathlib import Path

import yaml
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def main():
    params = yaml.safe_load(open("params.yaml"))["prepare"]

    digits = load_digits(as_frame=True)
    df = digits.frame  # 64 pixel columns + "target"

    train_df, test_df = train_test_split(
        df,
        test_size=params["test_size"],
        random_state=params["seed"],
        stratify=df["target"],
    )

    out = Path("data/processed")
    out.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out / "train.csv", index=False)
    test_df.to_csv(out / "test.csv", index=False)
    print(f"train={len(train_df)} test={len(test_df)}")


if __name__ == "__main__":
    main()
