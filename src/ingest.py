import pandas as pd
import os

def load_data(data_dir="data/raw"):
    label_df = pd.read_csv(os.path.join(data_dir, "actual.csv"))
    data_train = pd.read_csv(os.path.join(data_dir, "data_set_ALL_AML_train.csv"))
    data_test = pd.read_csv(os.path.join(data_dir, "data_set_ALL_AML_independent.csv"))
    return label_df, data_train, data_test

if __name__ == "__main__":
    label_df, data_train, data_test = load_data()
    print("shape of train", data_train.shape)
    print("shape of test", data_test.shape)
    print(label_df["cancer"].value_counts())
