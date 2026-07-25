import os
import joblib
import pandas as pd
import numpy as np

def load_trained_model(model_path):
    """
    Loads the saved model pipeline from disk.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}. Please run training first.")
    return joblib.load(model_path)

def calculate_optimal_preparation(predicted_demand, meal_type, special_event, variability_by_meal, safety_factor=0.8):
    """
    Calculates the recommended preparation quantity based on predicted demand and historical variability.
    
    Q_opt = predicted_demand + (adjusted_safety_factor * sigma_meal)
    
    Where:
    - sigma_meal is the historical standard deviation of consumption for that meal type.
    - adjusted_safety_factor is the user's safety_factor adjusted upwards for special events.
    """
    meal = str(meal_type).strip().capitalize()
    
    # Get standard deviation for the meal type
    sigma = variability_by_meal.get(meal, 15.0)  # fallback to 15 if not found
    
    # Adjust safety factor for special events (increase protection against stockout)
    adj_safety_factor = safety_factor
    if special_event:
        adj_safety_factor += 0.30
        
    # Calculate recommended quantity (rounded up)
    recommended = int(np.ceil(predicted_demand + (adj_safety_factor * sigma)))
    # Preparation cannot be negative
    recommended = max(1, recommended)
    
    return recommended, sigma, adj_safety_factor

def classify_risks(preparation_qty, predicted_demand, sigma):
    """
    Classifies shortage and waste risks based on preparation quantity,
    predicted demand, and historical variability (sigma).
    
    Shortage Risk:
    - High: Prepared < Predicted Demand (will run out)
    - Medium: Prepared >= Predicted Demand and Prepared < Predicted Demand + 0.5 * sigma (vulnerable to spikes)
    - Low: Prepared >= Predicted Demand + 0.5 * sigma (safe buffer)
    
    Waste Risk:
    - Low: Prepared <= Predicted Demand + 0.5 * sigma (leftovers minimized)
    - Medium: Prepared > Predicted Demand + 0.5 * sigma and Prepared <= Predicted Demand + 1.5 * sigma (leftovers manageable)
    - High: Prepared > Predicted Demand + 1.5 * sigma (excessive overproduction)
    """
    # Shortage classification
    if preparation_qty < predicted_demand:
        shortage_risk = "High"
    elif preparation_qty < predicted_demand + (0.5 * sigma):
        shortage_risk = "Medium"
    else:
        shortage_risk = "Low"
        
    # Waste classification
    if preparation_qty <= predicted_demand + (0.5 * sigma):
        waste_risk = "Low"
    elif preparation_qty <= predicted_demand + (1.5 * sigma):
        waste_risk = "Medium"
    else:
        waste_risk = "High"
        
    return shortage_risk, waste_risk

def predict_single_instance(model, input_dict, variability_by_meal, safety_factor=0.8):
    """
    Predicts food demand for a single set of features and calculates
    optimal preparation, expected leftovers, and risk levels.
    
    input_dict must contain:
    - day_of_week
    - meal_type
    - expected_customers
    - temperature
    - weather
    - special_event
    - is_weekend
    - prev_quantity_consumed
    - prev_quantity_wasted
    - rolling_avg_consumption
    - recent_demand_trend
    """
    # Convert input to DataFrame
    df_input = pd.DataFrame([input_dict])
    
    # Predict demand
    predicted_val = model.predict(df_input)[0]
    predicted_val = max(1.0, round(predicted_val, 1))
    
    # Calculate optimal preparation quantity
    meal_type = input_dict['meal_type']
    special_event = input_dict['special_event']
    
    recommended, sigma, adj_factor = calculate_optimal_preparation(
        predicted_val, meal_type, special_event, variability_by_meal, safety_factor
    )
    
    # Calculate expected leftovers if we prepare the recommended amount
    expected_leftover = max(0.0, round(recommended - predicted_val, 1))
    expected_waste_pct = round((expected_leftover / recommended * 100.0), 1) if recommended > 0 else 0.0
    
    # Classify risks for the recommended preparation
    shortage_risk, waste_risk = classify_risks(recommended, predicted_val, sigma)
    
    # Generate explanation text
    variability_desc = "high" if sigma > 20 else ("moderate" if sigma > 10 else "low")
    event_addition = " (with a +0.30 special event buffer adjustment)" if special_event else ""
    
    explanation = (
        f"SmartServe AI recommends preparing approximately {recommended} servings based on a predicted demand of "
        f"{predicted_val:.1f} servings and a historical {meal_type} demand variability of {sigma:.1f} ({variability_desc}). "
        f"A safety factor of {adj_factor:.2f}{event_addition} was applied. This balances the risk of running out of food "
        f"(Shortage Risk: {shortage_risk}) with the risk of unnecessary leftovers (Waste Risk: {waste_risk})."
    )
    
    return {
        "predicted_demand": predicted_val,
        "recommended_preparation": recommended,
        "expected_leftover": expected_leftover,
        "expected_waste_percentage": expected_waste_pct,
        "shortage_risk": shortage_risk,
        "waste_risk": waste_risk,
        "sigma": sigma,
        "safety_factor_used": adj_factor,
        "explanation": explanation
    }
