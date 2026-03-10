import sys
import os
sys.path.append(os.path.dirname(__file__))

from src.ingest import load_data
from src.preprocess import run_preprocessing
from src.train import (
    train_gaussian, bayesian_opt_gaussian,
    train_svm, bayesian_opt_svm,
    train_rf, bayesian_opt_rf,
    save_models,
)
from src.evaluate import evaluate_all


def main():
    print("\nSTEP 1: INGEST")
    label_df, data_train, data_test = load_data(data_dir="data/raw")

    print("\nSTEP 2: PREPROCESS")
    pca_data_train, pca_test_data, label_train, label_test = run_preprocessing(
        label_df, data_train, data_test
    )

    print("\nSTEP 3: TRAIN — Gaussian Process")
    train_gaussian(pca_data_train, label_train)
    bayesian_opt_gaussian(pca_data_train, label_train)

    print("\nSTEP 3: TRAIN — SVM")
    train_svm(pca_data_train, label_train)
    bayesian_opt_svm(pca_data_train, label_train)

    print("\nSTEP 3: TRAIN — Random Forest")
    train_rf(pca_data_train, label_train)
    bayesian_opt_rf(pca_data_train, label_train)

    print("\nSTEP 4: EVALUATE (best params from notebook)")
    gaussian_model, svm_model, rf_model = evaluate_all(
        pca_data_train, pca_test_data, label_train, label_test
    )

    print("\nSTEP 5: SAVE MODELS")
    save_models(gaussian_model, svm_model, rf_model)




if __name__ == "__main__":
    main()
