import os
import pandas as pd
import numpy as np
import datetime

def generate_realistic_data(output_path, num_days=180):
    np.random.seed(42)
    
    start_date = datetime.date(2026, 1, 1)
    meal_types = ['Breakfast', 'Lunch', 'Dinner']
    weather_types = ['Sunny', 'Cloudy', 'Rainy']
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Track historical consumption per meal type to compute rolling averages
    history_by_meal = {
        'Breakfast': [85.0, 90.0, 95.0],
        'Lunch': [190.0, 200.0, 205.0],
        'Dinner': [140.0, 150.0, 155.0]
    }
    
    # Keep track of previous overall values for simple lag
    prev_cons = 110.0
    prev_waste = 15.0
    
    data = []
    
    for day_idx in range(num_days):
        current_date = start_date + datetime.timedelta(days=day_idx)
        day_name = days_of_week[current_date.weekday()]
        is_weekend = day_name in ['Saturday', 'Sunday']
        weekend_val = 1 if is_weekend else 0
        
        # Temperature climbing slightly from Winter to Summer (18C to 33C)
        base_temp = 18.0 + (day_idx / num_days) * 15.0
        
        for meal in meal_types:
            # Weather probabilities: Sunny (60%), Cloudy (25%), Rainy (15%)
            weather = np.random.choice(weather_types, p=[0.60, 0.25, 0.15])
            
            # Temperature depends on weather
            temp_noise = np.random.normal(0, 1.8)
            if weather == 'Sunny':
                temp = base_temp + 3.0 + temp_noise
            elif weather == 'Cloudy':
                temp = base_temp + temp_noise
            else:  # Rainy
                temp = base_temp - 4.0 + temp_noise
            
            # Special Event: 5% chance overall, slightly higher on Fri/Sat
            event_prob = 0.15 if day_name in ['Friday', 'Saturday'] else 0.03
            special_event = 1 if np.random.random() < event_prob else 0
            
            # Base expected customer count depending on meal type
            if meal == 'Breakfast':
                base_cust = 100
                weekend_reduction = 0.40 if is_weekend else 0.0
            elif meal == 'Lunch':
                base_cust = 220
                weekend_reduction = 0.20 if is_weekend else 0.0
            else:  # Dinner
                base_cust = 180
                weekend_reduction = 0.15 if is_weekend else 0.0
                
            # Expected customers with noise
            expected_cust = base_cust * (1.0 - weekend_reduction)
            if special_event:
                expected_cust *= 1.30  # 30% increase for events
                
            expected_cust = int(np.random.normal(expected_cust, base_cust * 0.06))
            expected_cust = max(10, expected_cust)
            
            # Consumption / True demand calculation
            servings_per_customer = 0.94
            
            # Temperature effect: hot days (>31C) depress heavy meal consumption
            if temp > 31.0 and meal in ['Lunch', 'Dinner']:
                servings_per_customer -= 0.06
            # Rainy days increase traffic slightly as students avoid going off-campus
            if weather == 'Rainy':
                servings_per_customer += 0.04
            # Event increases excitement and portions
            if special_event:
                servings_per_customer += 0.03
                
            true_demand = expected_cust * servings_per_customer
            # Add realistic non-trivial noise (standard deviation ~ 7 servings)
            true_demand = np.random.normal(true_demand, 7.0)
            true_demand = max(5.0, round(true_demand, 1))
            
            # Feature engineering: Rolling Average of consumption for this specific meal
            # Average of the last 3 services of this meal type
            rolling_avg = float(np.mean(history_by_meal[meal][-3:]))
            
            # Feature engineering: Recent Demand Trend
            # difference between last meal's consumption and rolling average
            trend = float(history_by_meal[meal][-1] - rolling_avg)
            
            # Simulation of historical preparation by canteen staff (inefficient guesswork)
            # Staff often target expected customers but add a sloppy safety buffer (1.05 to 1.25)
            # Sometimes they ignore weekend drops (overproducing) or miss event spikes (underproducing)
            overprep_factor = np.random.uniform(1.06, 1.22)
            
            # Kitchen guess logic
            guess_prep = expected_cust * 0.94 * overprep_factor
            if meal == 'Dinner':
                guess_prep *= 1.04  # Dinner is often over-prepared
            
            # Lag effect: if previous waste was high, they might under-prepare
            if prev_waste > 20.0:
                guess_prep *= 0.92
                
            quantity_prepared = max(10.0, round(guess_prep, 1))
            
            # Actual consumed is capped at prepared quantity (can't consume what doesn't exist)
            if true_demand > quantity_prepared:
                quantity_consumed = quantity_prepared
                quantity_wasted = 0.0
            else:
                quantity_consumed = true_demand
                quantity_wasted = round(quantity_prepared - quantity_consumed, 1)
                
            waste_pct = round((quantity_wasted / quantity_prepared * 100.0), 1) if quantity_prepared > 0 else 0.0
            
            # Store in dataset
            data.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'day_of_week': day_name,
                'meal_type': meal,
                'expected_customers': expected_cust,
                'temperature': round(temp, 1),
                'weather': weather,
                'special_event': special_event,
                'is_weekend': weekend_val,
                'prev_quantity_consumed': round(prev_cons, 1),
                'prev_quantity_wasted': round(prev_waste, 1),
                'rolling_avg_consumption': round(rolling_avg, 1),
                'recent_demand_trend': round(trend, 1),
                'quantity_prepared': round(quantity_prepared, 1),
                'quantity_consumed': round(quantity_consumed, 1),
                'quantity_wasted': round(quantity_wasted, 1),
                'waste_percentage': waste_pct
            })
            
            # Update history and lag variables for next iterations
            history_by_meal[meal].append(quantity_consumed)
            prev_cons = quantity_consumed
            prev_waste = quantity_wasted
            
    df = pd.DataFrame(data)
    
    # Ensure folder directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[DATA GENERATION] Created realistic dataset with {len(df)} rows at: {output_path}")

if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_csv = os.path.join(project_root, 'data', 'food_waste_data.csv')
    generate_realistic_data(output_csv)
