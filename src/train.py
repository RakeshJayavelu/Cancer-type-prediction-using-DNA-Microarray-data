import numpy as np
import joblib
import os

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, DotProduct
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer


# ─── Gaussian Process ────────────────────────────────────────────────────────

def train_gaussian(pca_data_train, label_train):
    gaussian_model = GaussianProcessClassifier()

    rbf_kernel = RBF(length_scale=1.0)
    matern_kernel = Matern(length_scale=1.0, nu=1.5)
    dot_kernel = DotProduct(sigma_0=1.0)
    gaussian_classifier_kernels = [rbf_kernel, matern_kernel, dot_kernel]

    param_grid_gaussian = {
        "kernel": gaussian_classifier_kernels,
        "n_restarts_optimizer": [1, 5, 10],
        "max_iter_predict": [100, 1000],
    }

    grid_search_gaussian = GridSearchCV(
        estimator=gaussian_model, param_grid=param_grid_gaussian, cv=5
    )
    grid_search_gaussian.fit(pca_data_train, label_train)

    print("Best hyperparameters: ", grid_search_gaussian.best_params_)
    print("Best accuracy: ", grid_search_gaussian.best_score_)

    return grid_search_gaussian


def bayesian_opt_gaussian(pca_data_train, label_train):
    bo_kernels = [DotProduct(sigma_0=1.0), Matern(length_scale=1.0, nu=1.5), RBF(length_scale=1.0)]

    def bayesian_opt(kernel_bo):
        model_gp_bo = GaussianProcessClassifier(kernel=kernel_bo)
        param_space_gp = {
            "n_restarts_optimizer": Integer(1, 10),
            "max_iter_predict": Integer(100, 1000),
        }
        bayes_search_gp = BayesSearchCV(
            model_gp_bo,
            param_space_gp,
            n_iter=20,
            cv=5,
            n_jobs=-1,
            scoring="accuracy",
            random_state=42,
        )
        bayes_search_gp.fit(pca_data_train, label_train)
        return bayes_search_gp

    models_bo = []
    for i, kernel_i in enumerate(bo_kernels):
        i = bayesian_opt(kernel_i)
        models_bo.append(i)

    bo_accuracy = [x.best_score_ for x in models_bo]
    bestmodel_idx = bo_accuracy.index(max(bo_accuracy))

    print("best parameter:", models_bo[bestmodel_idx].best_params_)
    print("best kernel:", bo_kernels[bestmodel_idx])
    print("best accuracy: ", models_bo[bestmodel_idx].best_score_)

    return models_bo, bestmodel_idx


# ─── SVM ─────────────────────────────────────────────────────────────────────

def train_svm(pca_data_train, label_train):
    svm_clfr = SVC()

    param_grid_svm = {
        "C": [0.1, 1, 10, 100],
        "gamma": [0.1, 0.01],
        "kernel": ["rbf", "sigmoid"],
    }

    grid_search_svm = GridSearchCV(svm_clfr, param_grid_svm, cv=5)
    grid_search_svm_op = grid_search_svm.fit(pca_data_train, label_train)

    print("Best hyperparameters: ", grid_search_svm_op.best_params_)
    print("Best accuracy: ", grid_search_svm_op.best_score_)

    return grid_search_svm_op


def bayesian_opt_svm(pca_data_train, label_train):
    svm_bo_kernels = ["rbf", "sigmoid"]

    def bayesian_opt_svm_inner(svm_kernel_bo):
        model_svm_bo = SVC(kernel=svm_kernel_bo)
        param_space_gp = {
            "C": Real(0.1, 100.0),
            "gamma": Categorical([0.1, 0.01]),
        }
        bayes_search_svm = BayesSearchCV(
            model_svm_bo,
            param_space_gp,
            n_iter=20,
            cv=5,
            n_jobs=-1,
            scoring="accuracy",
            random_state=42,
        )
        bayes_search_svm.fit(pca_data_train, label_train)
        return bayes_search_svm

    svm_models_bo = []
    for i, kernel_i in enumerate(svm_bo_kernels):
        i = bayesian_opt_svm_inner(kernel_i)
        svm_models_bo.append(i)

    svm_bo_accuracy = [x.best_score_ for x in svm_models_bo]
    svm_bestmodel_idx = svm_bo_accuracy.index(max(svm_bo_accuracy))

    print("best parameter:", svm_models_bo[svm_bestmodel_idx].best_params_)
    print("best kernel:", svm_bo_kernels[svm_bestmodel_idx])
    print("best accuracy: ", max(svm_bo_accuracy))

    return svm_models_bo, svm_bestmodel_idx


# ─── Random Forest ───────────────────────────────────────────────────────────

def train_rf(pca_data_train, label_train):
    model_rf = RandomForestClassifier()

    param_grid_rf = {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, 20, None],
        "max_features": ["sqrt", "log2"],
    }

    grid_search_rf = GridSearchCV(model_rf, param_grid_rf, cv=5)
    grid_search_rf_op = grid_search_rf.fit(pca_data_train, label_train)

    print("Best hyperparameters: ", grid_search_rf_op.best_params_)
    print("Best accuracy: ", grid_search_rf_op.best_score_)

    return grid_search_rf_op


def bayesian_opt_rf(pca_data_train, label_train):
    rf_max_features = ["sqrt", "log2"]

    def bayesian_opt_rf_inner(rf_max_features):
        model_rf_bo = RandomForestClassifier(max_features=rf_max_features)
        param_space_rf = {
            "n_estimators": Integer(100, 300),
            "max_depth": Categorical([5, 10, 20, None]),
        }
        bayes_search_rf = BayesSearchCV(
            model_rf_bo,
            param_space_rf,
            n_iter=20,
            cv=5,
            n_jobs=-1,
            scoring="accuracy",
            random_state=42,
        )
        bayes_search_rf.fit(pca_data_train, label_train)
        return bayes_search_rf

    rf_models_bo = []
    for i, feature_i in enumerate(rf_max_features):
        i = bayesian_opt_rf_inner(feature_i)
        rf_models_bo.append(i)

    rf_bo_accuracy = [x.best_score_ for x in rf_models_bo]
    rf_bestmodel_idx = rf_bo_accuracy.index(max(rf_bo_accuracy))

    print("best parameter:", rf_models_bo[rf_bestmodel_idx].best_params_)
    print("best max_features:", rf_max_features[rf_bestmodel_idx])
    print("best accuracy: ", max(rf_bo_accuracy))

    return rf_models_bo, rf_bestmodel_idx


# ─── Save Models ─────────────────────────────────────────────────────────────

def save_models(gaussian_model, svm_model, rf_model, save_dir="data/processed"):
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(gaussian_model, os.path.join(save_dir, "gaussian_model.pkl"))
    joblib.dump(svm_model, os.path.join(save_dir, "svm_model.pkl"))
    joblib.dump(rf_model, os.path.join(save_dir, "rf_model.pkl"))
    print("Models saved to", save_dir)
