import pandas as pd
from datasets import Dataset

def load_data(path="data/IMDB Dataset.csv"):
    df = pd.read_csv(path)
    df = df.rename(columns={"review": "text", "sentiment": "label"})
    df["label"] = df["label"].map({"positive": 1, "negative": 0})
    return Dataset.from_pandas(df).train_test_split(test_size=0.2)
