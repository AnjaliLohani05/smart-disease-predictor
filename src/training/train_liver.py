import os
import sys
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.preprocess import preprocess_liver

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

CSV_PATH = os.path.join("data", "liver.csv")
MODEL_PATH = os.path.join("models", "liver.pkl")


def main():
    logging.info("Loading and preprocessing Liver Disease dataset …")
    X_train, X_test, y_train, y_test = preprocess_liver(CSV_PATH)

    logging.info("Training RandomForestClassifier …")
    # SVC(C=0.01) predicts only majority class on the imbalanced ILPD dataset;
    # RandomForest is more robust and was also highlighted in the notebook.
    model = RandomForestClassifier(
        criterion="gini",
        max_depth=5,
        n_estimators=100,
        class_weight="balanced",
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
        
    # Save Feature metadata for consistency
    models_dir = os.path.dirname(MODEL_PATH)
    with open(os.path.join(models_dir, "liver_features.pkl"), "wb") as f:
        pickle.dump(X_train.columns.tolist(), f)
        
    logging.info(f"Model and artifacts saved to {models_dir}")


if __name__ == "__main__":
    main()
