"""Train and compare classification models for churn prediction."""
import json
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)

from data_prep import load_data, build_preprocessor, get_train_test_split

MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "random_forest": RandomForestClassifier(
        n_estimators=300, max_depth=8, class_weight="balanced", random_state=42
    ),
}


def evaluate(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
    }


def main():
    df = load_data()
    X_train, X_test, y_train, y_test = get_train_test_split(df)

    results = {}
    best_name, best_score, best_pipeline = None, -1, None

    for name, clf in MODELS.items():
        pipe = Pipeline(steps=[("prep", build_preprocessor()), ("clf", clf)])
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        metrics = evaluate(y_test, y_pred, y_proba)
        results[name] = metrics
        print(f"{name}: {metrics}")

        if metrics["roc_auc"] > best_score:
            best_name, best_score, best_pipeline = name, metrics["roc_auc"], pipe

    joblib.dump(best_pipeline, "models/best_model.joblib")
    with open("models/metrics.json", "w") as f:
        json.dump({"results": results, "best_model": best_name}, f, indent=2)

    print(f"\nBest model: {best_name} (ROC-AUC={best_score})")
    print("Saved to models/best_model.joblib")


if __name__ == "__main__":
    main()
