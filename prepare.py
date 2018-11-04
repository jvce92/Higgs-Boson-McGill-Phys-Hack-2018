import pandas as pd
import numpy as np

def prepare_data(dataset_path):
    df = pd.read_csv(dataset_path)
    df["Label"] = df["Label"].apply(lambda l: l == "s") 

    X_train = df[df["KaggleSet"] == "t"].drop(["EventId", "Weight", "Label", "KaggleSet", "KaggleWeight"], axis=1)
    X_val = df[df["KaggleSet"] == "b"].drop(["EventId", "Weight", "Label", "KaggleSet", "KaggleWeight"], axis=1)
    X_test = df[df["KaggleSet"] == "v"].drop(["EventId", "Weight", "Label", "KaggleSet", "KaggleWeight"], axis=1)

    y_train = df[df["KaggleSet"] == "t"][["Label"]]
    y_val = df[df["KaggleSet"] == "b"][["Label"]]
    y_test = df[df["KaggleSet"] == "v"][["Label"]]

    W_train = df[df["KaggleSet"] == "t"]["Weight"]
    W_val = df[df["KaggleSet"] == "b"]["Weight"]
    W_test = df[df["KaggleSet"] == "v"]["Weight"]

    return df, X_train, y_train, W_train, X_val, y_val, W_val, X_test, y_test, W_test