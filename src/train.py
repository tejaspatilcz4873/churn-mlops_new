import argparse
import os
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from data_utils import load_data, prepare_xy, train_val_split


def build_pipeline():
    numeric_features = [
        "Age",
        "Tenure",
        "Usage Frequency",
        "Support Calls",
        "Payment Delay",
        "Total Spend",
        "Last Interaction",
    ]

    categorical_features = ["Gender", "Subscription Type", "Contract Length"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42, n_jobs=-1)),
        ]
    )

    return clf


def evaluate_model(model, X_val, y_val):
    preds = model.predict(X_val)
    probs = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "accuracy": float(accuracy_score(y_val, preds)),
        "precision": float(precision_score(y_val, preds, zero_division=0)),
        "recall": float(recall_score(y_val, preds, zero_division=0)),
        "f1": float(f1_score(y_val, preds, zero_division=0)),
    }

    if probs is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_val, probs))
        except Exception:
            metrics["roc_auc"] = None

    return metrics


def main(args):
    # Load data
    df = load_data(args.data_path)
    X, y = prepare_xy(df)
    X_train, X_val, y_train, y_val = train_val_split(X, y, test_size=0.2)

    # Build pipeline
    pipeline = build_pipeline()

    # Grid search
    param_grid = {
        "classifier__n_estimators": [50, 100],
        "classifier__max_depth": [None, 10],
    }

    grid = GridSearchCV(
        pipeline, param_grid, cv=3, scoring="f1", n_jobs=-1, verbose=1
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    metrics = evaluate_model(best_model, X_val, y_val)

    print("✅ Best params:", grid.best_params_)
    print("✅ Validation metrics:", metrics)

    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = Path(args.output_dir) / "churn_rf.joblib"
    joblib.dump(best_model, model_path)

    # Save metadata
    meta = {
        "model_path": str(model_path),
        "best_params": grid.best_params_,
        "metrics": metrics,
    }
    with open(Path(args.output_dir) / "model_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"✅ Model saved at: {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to input CSV data"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs", help="Directory to save model and metadata"
    )
    args = parser.parse_args()

    main(args)
