# SmartServe AI – Food Demand Prediction and Waste Reduction System

**SmartServe AI** is a complete, polished, hackathon-ready web application built for college canteens, hostels, cafeterias, event organizers, and restaurants. By leveraging Machine Learning, it replaces operational guesswork with predictive intelligence, helping providers estimate food demand, optimize preparation quantities, and minimize landfill food waste.

---

## 🍽️ The Problem & Impact

### The Problem
Food service managers face a daily challenge: preparing enough food to satisfy all customers while avoiding waste. Because registration and attendance fluctuate due to weather, weekends, and holidays, canteens often rely on **guesswork**. This yields a standard **10% to 25% overproduction margin** (surplus) to prevent stockouts, creating significant financial costs and carbon footprints.

### The Solution
SmartServe AI models actual demand from historical parameters. It uses a **Random Forest Regressor** to predict meal-by-meal consumption and recommends preparation limits with a dynamic safety buffer. 

### The Impact
- **Financial Savings**: Reduced food prep costs by lowering raw ingredient wastage.
- **Ecological Benefits**: Lower municipal landfill waste, minimizing methane emissions.
- **Operational Clarity**: Prediction-led planning for kitchen staff.

---

## 🛠️ Technology Stack

- **Python**: Core programming language.
- **Streamlit**: Web interface for creating dashboards and interactive simulators.
- **Scikit-learn**: Machine Learning pipeline (preprocessing, ColumnTransformer, Random Forest & Linear Regression estimators).
- **Pandas & NumPy**: Data processing and statistical manipulations.
- **Plotly**: Premium, interactive dashboard visualizations.
- **Joblib**: Serialization of trained pipeline objects.
- **CSV**: File format for data storage and custom uploads.

---

## 📁 Project Architecture

```
smartserve-ai/
│
├── app.py                      # Main Streamlit web application dashboard
├── requirements.txt            # Python dependencies list
├── README.md                   # Project documentation (this file)
│
├── data/
│   └── food_waste_data.csv     # Simulated baseline dataset (540 records)
│
├── models/
│   ├── model.pkl               # Saved Random Forest Pipeline (preprocessor + model)
│   └── model_metrics.json      # Model performance statistics and scatter points
│
└── src/
    ├── __init__.py             # Package initializer
    ├── generate_data.py        # Dataset simulator script (sequential simulation)
    ├── data_preprocessing.py   # Imputing, validating columns, and ColumnTransformer
    ├── train_model.py          # Model training, validation, and JSON metadata generation
    ├── prediction.py           # Single prediction pipeline & dynamic safety buffer rules
    ├── evaluation.py           # Evaluation calculations for metrics
    └── insights.py             # Rule-based dynamic insights generator
```

---

## 🚀 Getting Started

### 1. Installation

Ensure you have Python 3.8+ installed. Clone or copy the project files to your desktop, and navigate to the project root:

```bash
cd c:/Users/Sadaf/OneDrive/Desktop/sadaf_hackathon
```

Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

### 2. How to Run the Application

Start the Streamlit dashboard:

```bash
streamlit run app.py
```

*Note: On its very first run, if `models/model.pkl` is missing, the application will automatically run the simulator to create the dataset and train the ML models. It works 100% out-of-the-box!*

### 3. How to Retrain the Model

To retrain the model on the command line:

```bash
python src/train_model.py
```

Or visit the **📤 Data Upload & Retrain** page in the web interface, upload your custom CSV dataset, and click the retrain button to update the model in real time.

---

## 🧠 Machine Learning Approach

### 1. Features Used for Training
The model uses a collection of variables to understand student/customer behavior:
- **Day of week**: Captures weekend drops (students going home or sleeping in) and Friday swings.
- **Meal type**: Differentiates breakfast, lunch, and dinner consumption profiles.
- **Expected customers**: Represents event registrations, RSVP counts, or historical averages.
- **Temperature & Weather**: Sunny/rainy/cloudy variables (e.g., extreme heat dampens lunch; rain increases dining hall traffic).
- **Special event indicator**: Binary flag for campus festivals, holidays, or special banquets.
- **Lag variables (`prev_quantity_prepared`, `prev_quantity_consumed`, `prev_quantity_wasted`)**: Capture immediate momentum and canteen adjustments from the previous meal.

### 2. Preprocessing & Engineering
- Numerical columns are handled via `SimpleImputer(strategy='median')` and `StandardScaler()`.
- Categorical columns are processed via `SimpleImputer(strategy='most_frequent')` and `OneHotEncoder(handle_unknown='ignore')`.
- Preprocessing and estimators are bundled using Scikit-learn's `Pipeline` to prevent target leakage and simplify predictions.

### 3. Estimator Selection
- **Random Forest Regressor**: Primary model (n_estimators=100) to capture nonlinear interactions (e.g., Temperature × Meal Type).
- **Linear Regression**: Used as a baseline comparator.

### 4. Preparation Recommendation & Safety Buffers
To prevent stockouts (which are more critical operationally than a minor leftover), the app applies a dynamic safety buffer to the predicted demand:
- **Breakfast**: +8% buffer (more predictable)
- **Lunch**: +10% buffer
- **Dinner**: +12% buffer (higher variability)
- **Special Event**: Additional +5% buffer

$$\text{Recommendation} = \lceil \text{Predicted Demand} \times (1 + \text{Buffer}) \rceil$$

---

## 📈 Model Performance & Evaluation

The default trained model achieves the following benchmarks on the test split:
- **Mean Absolute Error (MAE)**: ~3.8 servings (predicts within ~4 servings of actual demand)
- **Root Mean Squared Error (RMSE)**: ~5.0 servings
- **Coefficient of Determination ($R^2$ Score)**: ~0.95 (explains 95% of demand variance)

---

## 💡 Future Scope

1. **Recipe Scaler Integration**: Automatically scale ingredient weights in the kitchen based on the recommended serving size.
2. **IoT Integration**: Integrate smart scale load sensors on garbage bins to capture waste data automatically.
3. **Portal Integrations**: Connect directly with student RSVP cards and class timetables to refine customer expectation signals.
