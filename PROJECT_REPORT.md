# Project Report: SmartServe AI
## Food Demand Prediction and Waste Reduction System

**Hackathon Domain:** Machine Learning  
**Project Title:** SmartServe AI  
**Deployment Status:** Live on Streamlit Community Cloud  
**Repository:** [github.com/sadaf2730/smartserve-ai](https://github.com/sadaf2730/smartserve-ai)  

---

## 📝 1. Executive Summary

**SmartServe AI** is an end-to-end Machine Learning web application designed to solve the dual challenges of food waste and meal shortages in large-scale catering services (college canteens, hostels, cafeterias, and event venues). 

Traditionally, kitchen managers rely on guesswork, over-preparing by **15% to 25%** to avoid stockouts. SmartServe AI replaces this heuristic with a **Random Forest Regressor** that forecasts precise food demand **BEFORE** preparation begins. By integrating historical demand variability ($\sigma_{meal}$), the system recommends an **Optimal Preparation Quantity** that minimizes both leftovers (waste) and shortages (hungry guests). 

On a 6-month historical dataset (540 records), the predictive engine achieved an **$R^2$ accuracy score of 0.9762** and a **Mean Absolute Error (MAE) of 6.38 servings**, demonstrating a dynamic waste reduction potential of **over 45%** with zero customer shortage risk.

---

## 🔴 2. Problem Statement

Food service operations suffer from severe inefficiencies due to demand uncertainty:
1. **The Overproduction Cost (Food Waste)**: Canteens routinely throw away 15% of cooked food. Landfilled food waste decomposes anaerobically, generating methane—a greenhouse gas 25x more potent than carbon dioxide. Financially, this discards thousands of dollars in raw ingredients and labor daily.
2. **The Underproduction Cost (Shortage)**: Preparing too little food leads to stockouts, resulting in student/customer dissatisfaction, long queues, and nutritional deficits.
3. **The Guesswork Loop**: Attendance is highly volatile, shifting due to weather changes, weekday/weekend schedules, calendar holidays, and special campus events. Catering managers lack tools to extract predictive patterns from these complex, overlapping variables.

---

## 🟢 3. Proposed Solution

SmartServe AI provides an intelligent, automated dashboard to manage and optimize food preparation:

```
Historical Food Data
        ↓
Data Cleaning & Feature Engineering
        ↓
Train/Test Split (80/20)
        ↓
Random Forest Regression (Demand Forecast)
        ↓
Variability Analysis (Optimal Serving Bounds)
        ↓
Dynamic Risk Profiling (Shortage vs. Waste Risk)
        ↓
Interactive Streamlit Dashboard & Simulator
```

* **Target Leakage Prevention**: We strictly isolate variables available *before* cooking. The model predicts actual customer demand (`quantity_consumed`) without using the current service's prepared limits as a feature.
* **Variability-Led Optimization**: Instead of static safety buffers, the system dynamically calculates standard deviations ($\sigma_{meal}$) for each meal type. It provides kitchen staff with an explainable recommendation matching the predicted demand plus a variability-scaled buffer, protecting against service spikes.
* **Shortage & Waste Risk Profiling**: Classifies operating plans into Low/Medium/High risk categories so managers understand the visual trade-offs of their cooking decisions in real-time.

---

## 📊 4. Data Layer & Feature Engineering

The system trains on a 6-month simulated sequential dataset (540 records) representing three daily services (Breakfast, Lunch, Dinner). The dataset incorporates realistic relationships, weekend patterns (e.g., student sleeping habits), and seasonal transitions:

### Training Features (Pre-Preparation Variables):
* **Expected Customer Registrations**: RSVP counts or ticket bookings.
* **Meal Type**: Breakfast, Lunch, or Dinner (distinct consumption profiles).
* **Day of Week & Weekend Indicator**: Monday–Sunday classifications capturing weekend attendance drops.
* **Weather & Temperature**: Simulates weather fluctuations (e.g., rainfall increases dining hall traffic).
* **Special Event Indicator**: Binary flag representing student festivals or campus holidays.
* **Temporal Lag Features**:
  * `prev_quantity_consumed` & `prev_quantity_wasted` (captures immediate demand swings).
  * `rolling_avg_consumption` (3-meal rolling average of the specific meal type).
  * `recent_demand_trend` (momentum difference relative to the rolling average).

---

## 🧠 5. Machine Learning & Model Performance

We trained a **Random Forest Regressor** (100 estimators) and compared it against a baseline **Linear Regression** model using an 80/20 train/test split. 

### Performance Metrics (On Test Split):

| Metric | Random Forest (SmartServe AI) | Linear Regression (Baseline) |
| :--- | :---: | :---: |
| **Mean Absolute Error (MAE)** | **6.38 servings** | 7.95 servings |
| **Root Mean Squared Error (RMSE)** | **8.14 servings** | 10.12 servings |
| **Coefficient of Determination ($R^2$)** | **0.9762** | 0.9632 |

* **Feature Importance**: Random Forest diagnostics reveal that **Expected Customers** holds the highest predictive weight (~68%), followed closely by **Rolling Average Consumption** (~18%) and **Meal Type** (~8%), proving that temporal lag features are vital for forecasting.

---

## ⚙️ 6. Optimal Preparation & Risk Logic

### 1. Optimal Preparation Quantity
To calculate recommended preparation servings ($Q_{opt}$), the prediction pipeline uses predicted demand ($D_{pred}$), the meal's historical standard deviation ($\sigma_{meal}$), and a configurable safety factor ($SF$, defaults to 0.8):

$$Q_{opt} = \lceil D_{pred} + (SF \times \sigma_{meal}) \rceil$$

If a special event is flagged, the safety factor is bumped by $+0.30$ to prevent stockouts during volatile guest spikes.

### 2. Dynamic Risk Classifications
For any target preparation quantity $Q$ chosen by a chef:
* **Shortage Risk**:
  * **High**: $Q < D_{pred}$ (guaranteed stockout)
  * **Medium**: $D_{pred} \le Q < D_{pred} + 0.5\sigma_{meal}$ (vulnerable to unexpected customer spikes)
  * **Low**: $Q \ge D_{pred} + 0.5\sigma_{meal}$ (safe buffer zone)
* **Waste Risk**:
  * **Low**: $Q \le D_{pred} + 0.5\sigma_{meal}$ (minimal leftovers)
  * **Medium**: $D_{pred} + 0.5\sigma_{meal} < Q \le D_{pred} + 1.5\sigma_{meal}$ (manageable waste)
  * **High**: $Q > D_{pred} + 1.5\sigma_{meal}$ (excessive overproduction)

---

## 🏠 7. Business & Ecological Impact

The landing page features a live **Waste Reduction Impact** panel that dynamically simulates performance over the 6-month historical logs:
* **Baseline Historical Waste Rate**: **13.9%** (overproduction due to guesswork).
* **AI-Optimized Waste Rate**: **7.2%** (wasted servings restricted to the optimal safety margin).
* **Potential Waste Reduction**: **~48.2% less food waste**.
* **Wasted Servings Rescued**: **3,524 servings** saved.
* **Estimated Financial Savings**: **$8,810.00** saved (assuming $2.50 per serving ingredient cost).
* **Carbon Offset equivalent**: **3,524 kg CO₂ equivalent** prevented from entering landfills (assuming 1 serving = 0.4 kg food; 1 kg waste = 2.5 kg CO₂).

---

## 💡 8. Future Scope

1. **Recipe Scaler Integration**: Automate back-of-house kitchen prep by converting recommended servings directly into exact ingredient weights (kilograms of rice, vegetables, meat).
2. **IoT Waste Bins Integration**: Connect with smart load cell scales placed on kitchen trash cans to automatically feed actual wastage metrics back into the system.
3. **Student RFID Portal Sync**: Link directly with campus card swipes and lecture timetables to gain real-time, hour-by-hour customer expectation signals.
