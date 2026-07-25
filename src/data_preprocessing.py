import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Required columns for the training dataset (to calculate KPIs and train models)
REQUIRED_COLUMNS = [
    'day_of_week', 'meal_type', 'expected_customers', 'temperature', 
    'weather', 'special_event', 'is_weekend', 'prev_quantity_consumed', 
    'prev_quantity_wasted', 'rolling_avg_consumption', 'recent_demand_trend',
    'quantity_prepared', 'quantity_consumed', 'quantity_wasted'
]

# Features that will be used in the ML model (excludes quantity_prepared to prevent leakage!)
FEATURE_COLUMNS = [
    'day_of_week', 'meal_type', 'expected_customers', 'temperature', 
    'weather', 'special_event', 'is_weekend', 'prev_quantity_consumed', 
    'prev_quantity_wasted', 'rolling_avg_consumption', 'recent_demand_trend'
]

TARGET_COLUMN = 'quantity_consumed'

def validate_dataset(df):
    """
    Validates if the DataFrame contains all required columns.
    Returns (is_valid, missing_columns)
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    return len(missing) == 0, missing

def load_and_clean_data(file_path):
    """
    Loads data from a CSV, validates columns, cleans numeric/categorical types,
    and returns a clean DataFrame.
    """
    df = pd.read_csv(file_path)
    
    # Validate
    is_valid, missing = validate_dataset(df)
    if not is_valid:
        raise ValueError(f"Missing required columns in dataset: {missing}")
        
    # Clean data: convert numeric fields to numeric type, handle potential malformed strings
    numeric_cols = [
        'expected_customers', 'temperature', 'special_event', 'is_weekend',
        'prev_quantity_consumed', 'prev_quantity_wasted', 'rolling_avg_consumption', 'recent_demand_trend',
        'quantity_prepared', 'quantity_consumed', 'quantity_wasted'
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # Calculate waste_percentage if missing or incorrect
    if 'waste_percentage' not in df.columns or df['waste_percentage'].isnull().any():
        df['waste_percentage'] = (df['quantity_wasted'] / df['quantity_prepared'] * 100.0).fillna(0.0)
        df['waste_percentage'] = df['waste_percentage'].round(1)
        
    return df

def get_preprocessing_pipeline():
    """
    Returns a ColumnTransformer preprocessing pipeline.
    This encodes categorical variables, scales numerical features, and handles missing values.
    """
    categorical_features = ['day_of_week', 'meal_type', 'weather']
    numerical_features = [
        'expected_customers', 'temperature', 'special_event', 'is_weekend',
        'prev_quantity_consumed', 'prev_quantity_wasted', 'rolling_avg_consumption', 'recent_demand_trend'
    ]
    
    # Pipeline for categorical columns
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Pipeline for numerical columns
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # drop any other columns (date, quantity_prepared, etc.)
    )
    
    return preprocessor
