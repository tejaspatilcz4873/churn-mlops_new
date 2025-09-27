import os
import joblib
import pandas as pd

# Path to the trained model (defaults to local file if env var not set)
MODEL_PATH = os.getenv("MODEL_PATH", "churn_rf.joblib")


def load_model():
    """Load the trained churn model."""
    model = joblib.load(MODEL_PATH)
    return model


def predict_single(model, input_dict):
    """
    Make a prediction for a single customer.
    
    Args:
        model: The trained model pipeline.
        input_dict: dict of feature_name -> value.
    Returns:
        dict with prediction and probability.
    """
    df = pd.DataFrame([input_dict])
    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0, 1] if hasattr(model, "predict_proba") else None
    return {
        "prediction": int(pred),
        "probability": float(proba) if proba is not None else None,
    }


if __name__ == "__main__":
    # Test the model locally
    model = load_model()

    sample = {
        "Age": 40,
        "Gender": "Female",
        "Tenure": 20,
        "Usage Frequency": 10,
        "Support Calls": 3,
        "Payment Delay": 5,
        "Subscription Type": "Standard",
        "Contract Length": "Monthly",
        "Total Spend": 400,
        "Last Interaction": 5,
    }

    result = predict_single(model, sample)
    print("Sample Prediction:", result)
