"""
Train Kidney Disease model.
Notebook: notebooks/kidney.ipynb
Model: RandomForestClassifier(criterion='entropy', max_depth=10,
       max_features='sqrt', min_samples_leaf=5, min_samples_split=2,
       n_estimators=400)
Output: models/kidney.pkl

Run from project root:
    python src/training/train_kidney.py
"""

import os
import sys
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.preprocess import preprocess_kidney

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

CSV_PATH = os.path.join("data", "kidney.csv")
MODEL_PATH = os.path.join("models", "kidney.pkl")


def main():
    logging.info("Loading and preprocessing Kidney Disease dataset …")
    (X_train, X_test, y_train, y_test), encoders, feature_cols, modes = preprocess_kidney(CSV_PATH)

    logging.info("Training RandomForestClassifier …")
    model = RandomForestClassifier(
        criterion="entropy",
        max_depth=10,
        max_features="sqrt",
        min_samples_leaf=5,
        min_samples_split=2,
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    logging.info(f"Test Accuracy: {acc * 100:.2f}%")
    logging.info(f"\n{classification_report(y_test, model.predict(X_test))}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    # Save Model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
        
    # Save Encoders, Modes and Feature metadata
    models_dir = os.path.dirname(MODEL_PATH)
    with open(os.path.join(models_dir, "kidney_encoders.pkl"), "wb") as f:
        pickle.dump(encoders, f)
    with open(os.path.join(models_dir, "kidney_modes.pkl"), "wb") as f:
        pickle.dump(modes, f)
    with open(os.path.join(models_dir, "kidney_features.pkl"), "wb") as f:
        pickle.dump(feature_cols, f)
        
    logging.info(f"Model and artifacts saved to {models_dir}")


if __name__ == "__main__":
    main()
