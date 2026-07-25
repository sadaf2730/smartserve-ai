import os
import json
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def load_metrics(metrics_path):
    """
    Loads saved model evaluation metrics from model_metrics.json.
    """
    if not os.path.exists(metrics_path):
        return None
        
    with open(metrics_path, 'r') as f:
        return json.load(f)

def evaluate_predictions(y_true, y_pred):
    """
    Calculates regression evaluation metrics on the fly.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    return {
        "mae": round(float(mae), 2),
        "rmse": round(float(rmse), 2),
        "r2": round(float(r2), 4)
    }
