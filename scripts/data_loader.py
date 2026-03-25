import pandas as pd
from pathlib import Path

TRAIN_PATH = Path("../dataset/raw/data/train")

def load_dataset():
    all_data = []

    for file in TRAIN_PATH.glob("*.csv"):
        df = pd.read_csv(file)

        if "benign" in file.name.lower():
            df["label"] = 0
        else:
            df["label"] = 1

        all_data.append(df)

    dataset = pd.concat(all_data, ignore_index=True)

    print("\nClass Distribution:")
    print(dataset["label"].value_counts())

    return dataset