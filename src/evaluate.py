from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def evaluate_all(pca_data_train, pca_test_data, label_train, label_test):

    # Gaussian Model
    final_gaussian_model = GaussianProcessClassifier(
        kernel=DotProduct(sigma_0=1), max_iter_predict=100, n_restarts_optimizer=1
    )
    final_gaussian_model.fit(pca_data_train, label_train)
    print(
        "Score on test dataset of Gaussian process classifier: ",
        final_gaussian_model.score(pca_test_data, label_test),
    )

    # SVM Classifier
    final_svm_model = SVC(C=1, gamma=0.1, kernel="sigmoid")
    final_svm_model.fit(pca_data_train, label_train)
    print(
        "Score on test dataset of SVM classifier: ",
        final_svm_model.score(pca_test_data, label_test),
    )

    # Random Forest Classifier
    final_rf_model = RandomForestClassifier(
        max_features="sqrt", max_depth=20, n_estimators=200
    )
    final_rf_model.fit(pca_data_train, label_train)
    print(
        "Score on test dataset of Random Forest classifier: ",
        final_rf_model.score(pca_test_data, label_test),
    )

    return final_gaussian_model, final_svm_model, final_rf_model
