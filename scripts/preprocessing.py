from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import numpy as np

def preprocess(df):

    # Separate labels
    y = df["label"]

    # Remove label column
    X = df.drop("label", axis=1)

    # -------- IMPORTANT FIX ----------
    # Keep only numeric hardware counter columns
    X = X.select_dtypes(include=[np.number])

    print("\nUsing numeric features:")
    print(X.columns)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    joblib.dump(scaler, "../models/scaler.pkl")

    return train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )