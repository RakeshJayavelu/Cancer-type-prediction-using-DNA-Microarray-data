import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import joblib
import os


def encode_labels(label_df):
    label_encdr = LabelEncoder()
    label_df["cancer"] = label_encdr.fit_transform(label_df["cancer"])
    label_df = label_df.set_index("patient")
    return label_df


def col_filter(df):
    call_list = df.filter(like="call").columns
    df = df.drop(call_list, axis=1)
    return df


def column_sorter(df):
    df.index = df.index.astype(int)
    df.sort_index(inplace=True)
    return df


def split_labels(label_df):
    label_train = label_df["cancer"][:38]
    label_test = label_df["cancer"][38:]
    return label_train, label_test


def reshape_data(data_train, data_test):
    data_train = data_train.T
    data_test = data_test.T

    data_train.columns = data_train.iloc[1]
    data_train = data_train.drop(["Gene Accession Number", "Gene Description"])

    data_test.columns = data_test.iloc[1]
    data_test = data_test.drop(["Gene Accession Number", "Gene Description"])

    return data_train, data_test


def scale_and_reduce(data_train, data_test, save_dir="data/processed"):
    scaling = StandardScaler()
    scaleddata_train = scaling.fit_transform(data_train)
    scaleddata_test = scaling.transform(data_test)

    pca = PCA(n_components=0.95)
    pca_data_train = pca.fit_transform(scaleddata_train)
    pca_test_data = pca.transform(scaleddata_test)

    print(pca_data_train.shape)
    print(pca_test_data.shape)

    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(scaling, os.path.join(save_dir, "scaler.pkl"))
    joblib.dump(pca, os.path.join(save_dir, "pca.pkl"))

    return pca_data_train, pca_test_data


def run_preprocessing(label_df, data_train, data_test):
    label_df = encode_labels(label_df)
    data_train = col_filter(data_train)
    data_test = col_filter(data_test)
    data_train, data_test = reshape_data(data_train, data_test)
    data_train = column_sorter(data_train)
    data_test = column_sorter(data_test)
    label_train, label_test = split_labels(label_df)
    pca_data_train, pca_test_data = scale_and_reduce(data_train, data_test)
    return pca_data_train, pca_test_data, label_train, label_test
