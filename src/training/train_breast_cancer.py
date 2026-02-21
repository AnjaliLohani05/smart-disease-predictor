"""
Train Breast Cancer model.
Notebook: notebooks/breast-cancer.ipynb
Model: SVC(C=10, gamma=0.01, probability=True)
Output: models/breast_cancer.pkl

Run from project root:
    python src/training/train_breast_cancer.py
"""

import os
import sys
import pickle
import logging
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.preprocess import preprocess_breast_cancer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

CSV_PATH = os.path.join("data", "breast_cancer.csv")
MODEL_PATH = os.path.join("models", "breast_cancer.pkl")


def main():
    logging.info("Loading and preprocessing Breast Cancer dataset …")
    X_train, X_test, y_train, y_test, scaler, feature_cols = preprocess_breast_cancer(CSV_PATH)

    logging.info("Training SVC …")
    model = SVC(C=10, gamma=0.01, probability=True)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    logging.info(f"Test Accuracy: {acc * 100:.2f}%")
    logging.info(f"\n{classification_report(y_test, model.predict(X_test))}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    # Save Model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
        
    # Save Scaler and Feature metadata
    models_dir = os.path.dirname(MODEL_PATH)
    with open(os.path.join(models_dir, "breast_cancer_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(models_dir, "breast_cancer_features.pkl"), "wb") as f:
        pickle.dump(feature_cols, f)
        
    logging.info(f"Model and artifacts saved to {models_dir}")


if __name__ == "__main__":
    main()
