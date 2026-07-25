import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

# Import preprocessing
from data_preprocessing import load_and_clean_data, get_preprocessing_pipeline, FEATURE_COLUMNS, TARGET_COLUMN

def train_and_evaluate(data_path, models_dir):
    """
    Trains Random Forest and Linear Regression models, evaluates them,
    calculates meal-type demand variability, and saves model and metadata.
    """
    print(f"[MODEL TRAINING] Loading data from {data_path}...")
    df = load_and_clean_data(data_path)
    
    # Split features and target
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    
    # Split into train and test sets (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Preprocessor
    preprocessor = get_preprocessing_pipeline()
    
    # Models
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    lr_regressor = LinearRegression()
    
    # Pipelines
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', rf_regressor)
    ])
    
    lr_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', lr_regressor)
    ])
    
    # Fit models
    print("[MODEL TRAINING] Fitting Random Forest Regressor...")
    rf_pipeline.fit(X_train, y_train)
    
    print("[MODEL TRAINING] Fitting Linear Regression baseline...")
    lr_pipeline.fit(X_train, y_train)
    
    # Predict
    rf_preds = rf_pipeline.predict(X_test)
    lr_preds = lr_pipeline.predict(X_test)
    
    # Evaluate RF
    rf_mae = mean_absolute_error(y_test, rf_preds)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
    rf_r2 = r2_score(y_test, rf_preds)
    
    # Evaluate LR
    lr_mae = mean_absolute_error(y_test, lr_preds)
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))
    lr_r2 = r2_score(y_test, lr_preds)
    
    print("\n--- RANDOM FOREST METRICS ---")
    print(f"MAE:  {rf_mae:.2f} servings")
    print(f"RMSE: {rf_rmse:.2f} servings")
    print(f"R2:   {rf_r2:.4f}")
    
    # Calculate historical consumption variability (std dev) by meal type
    # This represents the historical demand variability used for optimal prep calculations
    variability_by_meal = {}
    for m in ['Breakfast', 'Lunch', 'Dinner']:
        meal_df = df[df['meal_type'].str.strip().str.capitalize() == m]
        if not meal_df.empty and len(meal_df) > 1:
            std_val = meal_df['quantity_consumed'].std()
            if pd.isna(std_val) or std_val == 0:
                std_val = 12.0 if m == 'Breakfast' else (25.0 if m == 'Lunch' else 18.0)
        else:
            # Fallbacks
            std_val = 12.0 if m == 'Breakfast' else (25.0 if m == 'Lunch' else 18.0)
        variability_by_meal[m] = round(float(std_val), 2)
        
    print(f"[MODEL TRAINING] Calculated demand variability by meal type: {variability_by_meal}")
    
    # Get feature names from ColumnTransformer
    trained_preprocessor = rf_pipeline.named_steps['preprocessor']
    feature_names = trained_preprocessor.get_feature_names_out()
    
    # Clean up names for display
    clean_names = []
    for name in feature_names:
        name = name.replace('num__', '').replace('cat__', '')
        clean_names.append(name)
        
    importances = rf_pipeline.named_steps['regressor'].feature_importances_
    
    # Group feature importances for displaying in charts
    feature_importance_list = [
        {"feature": name, "importance": float(imp)}
        for name, imp in zip(clean_names, importances)
    ]
    feature_importance_list = sorted(feature_importance_list, key=lambda x: x['importance'], reverse=True)
    
    # Build complete metadata object
    metrics_metadata = {
        "model_name": "Random Forest Regressor",
        "training_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "dataset_size": len(df),
        "test_size": len(X_test),
        "rf_metrics": {
            "mae": round(float(rf_mae), 2),
            "rmse": round(float(rf_rmse), 2),
            "r2": round(float(rf_r2), 4)
        },
        "lr_metrics": {
            "mae": round(float(lr_mae), 2),
            "rmse": round(float(lr_rmse), 2),
            "r2": round(float(lr_r2), 4)
        },
        "variability_by_meal": variability_by_meal,
        "feature_importances": feature_importance_list,
        "actual_vs_predicted": {
            "actual": y_test.tolist(),
            "predicted": [round(float(p), 1) for p in rf_preds]
        }
    }
    
    # Save the pipeline
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'model.pkl')
    joblib.dump(rf_pipeline, model_path)
    print(f"[MODEL TRAINING] Saved Random Forest Pipeline to: {model_path}")
    
    # Save metrics metadata
    metrics_path = os.path.join(models_dir, 'model_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_metadata, f, indent=4)
    print(f"[MODEL TRAINING] Saved metrics metadata to: {metrics_path}")
    
    return metrics_metadata

if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_csv = os.path.join(project_root, 'data', 'food_waste_data.csv')
    models_path = os.path.join(project_root, 'models')
    train_and_evaluate(data_csv, models_path)
