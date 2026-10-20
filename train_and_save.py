"""
train_and_save.py
=================
Automated ML Training & Serialization Pipeline for Ford Car Price Prediction.
Generates genuine evaluation metrics, test residuals, and pipeline artifacts.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def train_and_export():
    dataset_path = "ford.csv"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"'{dataset_path}' not found in the root directory.")

    print("[1/5] Loading and cleaning raw dataset...")
    df = pd.read_csv(dataset_path)

    # Clean whitespace in categorical strings (e.g. ' Fiesta' -> 'Fiesta')
    for col in ['model', 'transmission', 'fuelType']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Drop impossible registration year outliers if present (e.g., year 2060)
    df = df[(df['year'] >= 1990) & (df['year'] <= 2024)]
    df = df[df['engineSize'] > 0.0]  # Remove 0.0L engine data anomalies
    df = df.dropna().reset_index(drop=True)

    X = df.drop(columns=['price'])
    y = df['price']

    categorical_features = ['model', 'transmission', 'fuelType']
    numerical_features = ['year', 'mileage', 'tax', 'mpg', 'engineSize']

    print("[2/5] Creating preprocessing and model pipeline...")
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
        ]
    )

    # Using RandomForestRegressor for high non-linear valuation fidelity (R2 > 0.90)
    regressor = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', regressor)
    ])

    print("[3/5] Splitting dataset (80% Train, 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    print("[4/5] Training model pipeline...")
    model_pipeline.fit(X_train, y_train)

    print("[5/5] Evaluating performance on test partition...")
    y_pred = model_pipeline.predict(X_test)

    r2 = float(r2_score(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    
    print(f" -> Genuine Test R² Score : {r2:.4f}")
    print(f" -> Mean Absolute Error   : £{mae:,.2f}")
    print(f" -> Root Mean Sq. Error   : £{rmse:,.2f}")

    # Build evaluation sample for plotting Actual vs. Predicted (subsample 600 points for speed)
    eval_df = pd.DataFrame({
        'Actual': y_test.values,
        'Predicted': y_pred
    }).sample(n=min(600, len(y_test)), random_state=42)

    os.makedirs("artifacts", exist_ok=True)

    # Save Pipeline artifact
    joblib.dump(model_pipeline, "artifacts/model_pipeline.joblib")

    # Save Metadata and Metrics
    metadata = {
        "model_name": "Random Forest Ensemble Regressor",
        "n_train_samples": int(len(X_train)),
        "n_test_samples": int(len(X_test)),
        "metrics": {
            "r2_score": round(r2, 4),
            "mae": round(mae, 2),
            "rmse": round(rmse, 2)
        },
        "features": {
            "numerical": numerical_features,
            "categorical": categorical_features,
            "models": sorted(df['model'].unique().tolist()),
            "transmissions": sorted(df['transmission'].unique().tolist()),
            "fuel_types": sorted(df['fuelType'].unique().tolist())
        },
        "dataset_stats": {
            "total_records": int(len(df)),
            "mean_price": round(float(df['price'].mean()), 2),
            "median_price": round(float(df['price'].median()), 2),
            "min_price": round(float(df['price'].min()), 2),
            "max_price": round(float(df['price'].max()), 2),
            "avg_mileage": round(float(df['mileage'].mean()), 1),
            "avg_mpg": round(float(df['mpg'].mean()), 1)
        },
        "eval_sample": eval_df.to_dict(orient="records")
    }

    with open("artifacts/metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print("\n✓ Pipeline and metadata exported successfully to 'artifacts/'")

if __name__ == "__main__":
    train_and_export()