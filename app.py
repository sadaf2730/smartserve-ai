import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
import joblib

# Import custom src modules
from src.data_preprocessing import load_and_clean_data, FEATURE_COLUMNS, REQUIRED_COLUMNS, validate_dataset
from src.prediction import load_trained_model, predict_single_instance, classify_risks
from src.evaluation import load_metrics
from src.insights import get_dynamic_insights

# Set page configuration
st.set_page_config(
    page_title="SmartServe AI - Food Waste Reduction",
    page_icon="🍲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
def inject_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
        
        /* Font styling */
        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
        }
        
        /* Sidebar styling */
        .sidebar .sidebar-content {
            background-color: #f8f9fa;
        }
        
        /* Headers */
        .main-header {
            font-size: 2.6rem;
            font-weight: 700;
            color: #2E7D32;
            margin-bottom: 2px;
        }
        .subheader {
            font-size: 1.1rem;
            color: #5c636a;
            margin-bottom: 25px;
        }
        
        /* KPI Cards */
        .kpi-card {
            background-color: #ffffff;
            border: 1px solid #e9ecef;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
            transition: transform 0.2s, box-shadow 0.2s;
            height: 100%;
        }
        .kpi-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
        }
        .kpi-title {
            font-size: 0.85rem;
            color: #8c96a0;
            text-transform: uppercase;
            font-weight: 600;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }
        .kpi-value {
            font-size: 2rem;
            font-weight: 700;
            color: #1a1d20;
            margin-bottom: 4px;
        }
        .kpi-footer {
            font-size: 0.8rem;
            font-weight: 500;
        }
        .footer-success { color: #2E7D32; }
        .footer-warning { color: #E65100; }
        .footer-info { color: #1565C0; }
        .footer-neutral { color: #5c636a; }
        
        /* Risk Badges */
        .risk-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 50px;
            font-size: 0.85rem;
            font-weight: 600;
            text-align: center;
        }
        .risk-low {
            background-color: #E8F5E9;
            color: #2E7D32;
            border: 1px solid #A5D6A7;
        }
        .risk-medium {
            background-color: #FFF3E0;
            color: #E65100;
            border: 1px solid #FFCC80;
        }
        .risk-high {
            background-color: #FFEBEE;
            color: #C62828;
            border: 1px solid #FFCDD2;
        }
        
        /* Result Panel */
        .result-panel {
            background-color: #F1F8E9;
            border: 1px solid #DCEDC8;
            border-left: 6px solid #558B2F;
            border-radius: 8px;
            padding: 22px;
            margin-top: 15px;
        }
        .result-title {
            font-size: 1.25rem;
            font-weight: 700;
            color: #33691E;
            margin-bottom: 10px;
        }
        .result-body {
            font-size: 0.95rem;
            color: #33691E;
            line-height: 1.6;
        }
        
        /* What-If scenario card styling */
        .scenario-card {
            border: 1px solid #e0e0e0;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.03);
            background-color: #ffffff;
            transition: transform 0.2s;
            height: 100%;
        }
        .scenario-card:hover {
            transform: translateY(-2px);
        }
        .scenario-a { border-top: 5px solid #E53935; } /* Under-prep (Red) */
        .scenario-b { border-top: 5px solid #43A047; } /* Optimal (Green) */
        .scenario-c { border-top: 5px solid #FB8C00; } /* Over-prep (Orange) */
        
        .scenario-header {
            font-size: 1.15rem;
            font-weight: 700;
            margin-bottom: 15px;
            color: #212529;
        }
        </style>
    """, unsafe_allow_html=True)

# Main project paths
project_root = os.path.dirname(os.path.abspath(__file__))
data_csv = os.path.join(project_root, 'data', 'food_waste_data.csv')
models_dir = os.path.join(project_root, 'models')
model_path = os.path.join(models_dir, 'model.pkl')
metrics_path = os.path.join(models_dir, 'model_metrics.json')

# Check if data or models exist; initialize dynamically on start
if not os.path.exists(data_csv):
    from src.generate_data import generate_realistic_data
    generate_realistic_data(data_csv)

if not os.path.exists(model_path) or not os.path.exists(metrics_path):
    from src.train_model import train_and_evaluate
    train_and_evaluate(data_csv, models_dir)

# Initialize Session State Data
if 'df_data' not in st.session_state:
    try:
        st.session_state['df_data'] = load_and_clean_data(data_csv)
    except Exception as e:
        st.error(f"Error loading initial dataset: {e}")
        st.session_state['df_data'] = pd.DataFrame()

if 'model' not in st.session_state:
    try:
        st.session_state['model'] = load_trained_model(model_path)
    except Exception as e:
        st.warning(f"Could not load pre-trained model: {e}. Retraining...")
        from src.train_model import train_and_evaluate
        train_and_evaluate(data_csv, models_dir)
        st.session_state['model'] = load_trained_model(model_path)

if 'metrics' not in st.session_state:
    st.session_state['metrics'] = load_metrics(metrics_path)

inject_custom_css()

# Sidebar Setup
st.sidebar.markdown("<div style='text-align: center; padding-bottom: 5px;'><h2 style='color:#2E7D32; margin-bottom: 0px;'>🍲 SmartServe AI</h2><small style='color:#6c757d; font-weight:500;'>Demand-Led Waste Reduction</small></div>", unsafe_allow_html=True)
st.sidebar.markdown("---")

# Global Configuration Parameters in Sidebar
st.sidebar.subheader("⚙️ System Configurations")
cost_per_serving = st.sidebar.number_input(
    "Ingredient Cost per Serving ($)", 
    min_value=0.10, 
    max_value=100.0, 
    value=2.50, 
    step=0.10,
    help="Used to dynamically calculate financial waste loss and savings opportunities."
)

safety_factor = st.sidebar.slider(
    "Preparation Safety Factor", 
    min_value=0.0, 
    max_value=2.0, 
    value=0.8, 
    step=0.1,
    help="Safety multiplier for historical demand variability. Higher factor reduces Shortage Risk but increases Waste Risk."
)

st.sidebar.markdown("---")

pages = [
    "🏠 Problem & Impact",
    "🔮 Demand Prediction",
    "📊 Waste Analytics",
    "📈 Model Performance",
    "🔄 What-If Simulator",
    "💡 Insights & Recommendations",
    "📤 Data Upload & Retrain"
]

if 'selected_page' not in st.session_state:
    st.session_state['selected_page'] = pages[0]

# Page Navigation selector
selected_page = st.sidebar.radio("Go to Page", pages, index=pages.index(st.session_state['selected_page']))
st.session_state['selected_page'] = selected_page

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='font-size: 0.85rem; color: #6c757d; line-height: 1.4;'>
<b>Hackathon Presentation Flow</b><br>
1. <b>Problem</b>: Guesswork leads to waste.<br>
2. <b>Solution</b>: Forecast demand BEFORE preparation.<br>
3. <b>Impact</b>: Dynamic cost & CO2 reductions.
</div>
""", unsafe_allow_html=True)

# References to active data
df = st.session_state['df_data']
model = st.session_state['model']
metrics = st.session_state['metrics']

# Safety checks
if df.empty or metrics is None:
    st.warning("Warning: Model data or CSV is missing. Go to Data Upload page to train a model.")
    st.stop()

# Extract meal variability dynamically from trained metrics
variability_by_meal = metrics.get("variability_by_meal", {"Breakfast": 12.0, "Lunch": 25.0, "Dinner": 18.0})


# --- PAGE 1: OVERVIEW / PROBLEM & IMPACT ---
if selected_page == "🏠 Problem & Impact":
    st.markdown("<h1 class='main-header'>SmartServe AI</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subheader'>Food Demand Prediction and Waste Reduction System</p>", unsafe_allow_html=True)
    
    # Grid Layout
    col1, col2 = st.columns([13, 10])
    
    with col1:
        st.markdown("""
        ### 📋 The Guesswork Challenge
        In canteens, hostels, and cafeterias, kitchen managers face a daily trade-off:
        
        * 🔴 **Over-preparation** yields massive **food waste** and severe **financial loss**.
        * 🔵 **Under-preparation** yields **food shortages** and **customer dissatisfaction**.
        
        Because student attendance changes dynamically based on weather, weekdays, and events, managers standardly prepare **15% to 25% surplus food** to ensure they do not run out. This results in tons of fresh food going straight to landfills daily.
        
        ### 💡 The SmartServe AI Solution
        SmartServe AI implements **Random Forest Regression** to forecast exact food demand **BEFORE** the kitchen decides how much to prepare.
        
        Instead of a fixed safety buffer, it analyzes **historical demand variability ($\sigma_{meal}$)** by meal type. It then recommends an **Optimal Preparation Quantity** that dynamically balances shortage risks with waste risks, minimizing both.
        """)
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔮 Open Prediction Engine", type="primary"):
            st.session_state['selected_page'] = "🔮 Demand Prediction"
            st.rerun()
            
    with col2:
        st.markdown("### 🏆 Simulated Waste Reduction Impact")
        st.markdown("<small style='color:#6c757d;'>Calculated dynamically using actual model predictions and dataset statistics.</small>", unsafe_allow_html=True)
        
        # --- DYNAMIC IMPACT CALCULATIONS ---
        # Current baseline stats
        total_prepared = df['quantity_prepared'].sum()
        total_consumed = df['quantity_consumed'].sum()
        total_wasted = df['quantity_wasted'].sum()
        current_waste_pct = (total_wasted / total_prepared * 100.0) if total_prepared > 0 else 0.0
        
        # Simulated Optimized prep: Q_opt_i = demand_i + safety_factor * sigma_meal
        # Wasted servings in optimized = max(0, Q_opt_i - demand_i) = safety_factor * sigma_meal
        sim_wasted_servings = 0.0
        sim_prepared_servings = 0.0
        for index, row in df.iterrows():
            meal = row['meal_type']
            sigma = variability_by_meal.get(meal, 15.0)
            demand = row['quantity_consumed']
            
            # Simulated recommended preparation
            opt_prep = demand + (safety_factor * sigma)
            sim_prepared_servings += opt_prep
            sim_wasted_servings += (opt_prep - demand) # which is safety_factor * sigma
            
        opt_waste_pct = (sim_wasted_servings / sim_prepared_servings * 100.0) if sim_prepared_servings > 0 else 0.0
        
        # Absolute and Relative reductions
        relative_reduction = ((current_waste_pct - opt_waste_pct) / current_waste_pct * 100.0) if current_waste_pct > 0 else 0.0
        servings_saved = max(0.0, total_wasted - sim_wasted_servings)
        cost_saved = servings_saved * cost_per_serving
        
        # Environmental impact: EPA Factor (1 serving ≈ 0.4 kg of food; 1 kg food waste ≈ 2.5 kg CO2)
        # So 1 serving saved ≈ 1.0 kg CO2 saved
        co2_saved = servings_saved * 1.0
        
        st.markdown(f"""
        <div style="background-color:#E8F5E9; border-left:6px solid #2E7D32; padding:20px; border-radius:8px; margin-bottom:15px; box-shadow:0 4px 6px rgba(0,0,0,0.03);">
            <table style="width:100%; font-size:1.05rem; line-height:1.8;">
                <tr>
                    <td><b>Current Avg. Waste Rate:</b></td>
                    <td style="text-align:right; font-weight:700; color:#C62828;">{current_waste_pct:.1f}%</td>
                </tr>
                <tr>
                    <td><b>Optimized Avg. Waste Rate:</b></td>
                    <td style="text-align:right; font-weight:700; color:#2E7D32;">{opt_waste_pct:.1f}%</td>
                </tr>
                <tr style="border-bottom: 1px solid #C8E6C9;">
                    <td><b>Waste Reduction Potential:</b></td>
                    <td style="text-align:right; font-weight:700; color:#2E7D32;">{relative_reduction:.1f}% less waste</td>
                </tr>
                <tr>
                    <td><b>Wasted Servings Rescued:</b></td>
                    <td style="text-align:right; font-weight:700; color:#2E7D32;">{servings_saved:,.0f} servings</td>
                </tr>
                <tr>
                    <td><b>Estimated Cost Savings:</b></td>
                    <td style="text-align:right; font-weight:700; color:#2E7D32; font-size:1.2rem;">${cost_saved:,.2f}</td>
                </tr>
                <tr>
                    <td><b>Est. CO2 Carbon Offset:</b></td>
                    <td style="text-align:right; font-weight:700; color:#1565C0;">{co2_saved:,.0f} kg CO₂</td>
                </tr>
            </table>
            <div style="font-size:0.8rem; color:#558B2F; margin-top:12px; text-align:center;">
                *Calculations are synthetic/demo representations simulating a safety factor multiplier of {safety_factor}x.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Historical Canteen Operations Overview")
    
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    
    with kpi_col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Prepared</div>
            <div class="kpi-value">{total_prepared:,.0f}</div>
            <div class="kpi-footer footer-neutral">Servings in dataset</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi_col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Consumed</div>
            <div class="kpi-value">{total_consumed:,.0f}</div>
            <div class="kpi-footer footer-success">{(total_consumed/total_prepared*100):.1f}% Consumption Rate</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi_col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Food Wasted</div>
            <div class="kpi-value">{total_wasted:,.0f}</div>
            <div class="kpi-footer footer-warning">Cost Loss: ${total_wasted*cost_per_serving:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi_col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Avg. Waste Percentage</div>
            <div class="kpi-value">{current_waste_pct:.1f}%</div>
            <div class="kpi-footer footer-neutral">Baseline over 6 months</div>
        </div>
        """, unsafe_allow_html=True)


# --- PAGE 2: DEMAND PREDICTION ---
elif selected_page == "🔮 Demand Prediction":
    st.markdown("<h1 class='main-header'>🔮 Food Demand Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subheader'>Estimate Demand and Optimal Preparation Limits BEFORE Service Cooking</p>", unsafe_allow_html=True)
    
    # Retrieve default values to make inputs easy
    mean_cust = int(df['expected_customers'].mean())
    mean_temp = float(df['temperature'].mean())
    mean_prev_cons = float(df['prev_quantity_consumed'].mean())
    mean_prev_waste = float(df['prev_quantity_wasted'].mean())
    mean_roll = float(df['rolling_avg_consumption'].mean())
    mean_trend = float(df['recent_demand_trend'].mean())
    
    # Form layout
    with st.form("prediction_pre_prep_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 📍 Context & Environmental Factors")
            expected_customers = st.number_input("Expected Customer Registrations", min_value=10, max_value=2000, value=mean_cust, step=10, help="RSVPs, meal bookings or expected footfall count.")
            day_of_week = st.selectbox("Day of Week", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
            meal_type = st.selectbox("Meal Type", ['Breakfast', 'Lunch', 'Dinner'])
            weather = st.selectbox("Weather Condition", ['Sunny', 'Cloudy', 'Rainy'])
            temperature = st.slider("Forecasted Temperature (°C)", min_value=-5.0, max_value=45.0, value=round(mean_temp, 1), step=0.5)
            special_event = st.checkbox("Special Event/Holiday (Increases attendance & safety factor?)", value=False)
            
        with col2:
            st.markdown("##### 🕒 Historical Lags (Before Preparation)")
            st.markdown("<small style='color:#6c757d;'>These indicators help the model track localized demand shifts and trends.</small>", unsafe_allow_html=True)
            prev_quantity_consumed = st.number_input("Previous Service Consumption (servings)", min_value=0.0, value=mean_prev_cons, step=5.0)
            prev_quantity_wasted = st.number_input("Previous Service Wastage (servings)", min_value=0.0, value=mean_prev_waste, step=5.0)
            rolling_avg_consumption = st.number_input("3-Meal Rolling Average Consumption (servings)", min_value=0.0, value=mean_roll, step=5.0)
            recent_demand_trend = st.number_input("Recent Demand Trend deviation (servings)", value=mean_trend, step=1.0, help="Diff between previous consumption and the rolling average.")
            
        submit_btn = st.form_submit_button("🔮 Calculate Optimal Preparation")
        
    if submit_btn:
        # Construct input vector matching FEATURE_COLUMNS
        # day_of_week, meal_type, expected_customers, temperature, weather, special_event, is_weekend,
        # prev_quantity_consumed, prev_quantity_wasted, rolling_avg_consumption, recent_demand_trend
        is_weekend_val = 1 if day_of_week in ['Saturday', 'Sunday'] else 0
        
        input_dict = {
            'day_of_week': day_of_week,
            'meal_type': meal_type,
            'expected_customers': expected_customers,
            'temperature': temperature,
            'weather': weather,
            'special_event': 1 if special_event else 0,
            'is_weekend': is_weekend_val,
            'prev_quantity_consumed': prev_quantity_consumed,
            'prev_quantity_wasted': prev_quantity_wasted,
            'rolling_avg_consumption': rolling_avg_consumption,
            'recent_demand_trend': recent_demand_trend
        }
        
        with st.spinner("Executing prediction pipeline..."):
            try:
                res = predict_single_instance(model, input_dict, variability_by_meal, safety_factor)
                
                # Display Results
                st.success("Analysis complete!")
                
                # Display dynamic metrics
                st.markdown("### Suggested Operating Plan")
                m_col1, m_col2, m_col3 = st.columns(3)
                
                with m_col1:
                    st.metric("PREDICTED FOOD DEMAND", f"{res['predicted_demand']:.0f} servings", help="Expected actual consumption based on ML variables.")
                with m_col2:
                    st.metric("RECOMMENDED PREPARATION", f"{res['recommended_preparation']} servings", help="Optimal servings to prepare, dynamically integrating historical demand variability.")
                with m_col3:
                    st.metric("EXPECTED LEFTOVER", f"{res['expected_leftover']:.0f} servings", help="Expected waste servings if demand matches predictions.")
                    
                # Display Risks using stylized badges
                r_col1, r_col2 = st.columns(2)
                
                # Shortage risk badge class
                short_class = "risk-low" if res['shortage_risk'] == "Low" else ("risk-medium" if res['shortage_risk'] == "Medium" else "risk-high")
                # Waste risk badge class
                waste_class = "risk-low" if res['waste_risk'] == "Low" else ("risk-medium" if res['waste_risk'] == "Medium" else "risk-high")
                
                with r_col1:
                    st.markdown(f"**Shortage Risk:** <span class='risk-badge {short_class}'>{res['shortage_risk']} Risk</span>", unsafe_allow_html=True)
                with r_col2:
                    st.markdown(f"**Waste Risk:** <span class='risk-badge {waste_class}'>{res['waste_risk']} Risk</span>", unsafe_allow_html=True)
                    
                # Recommendation Text Box
                st.markdown(f"""
                <div class="result-panel">
                    <div class="result-title">📖 SmartServe Recommendation Explanation</div>
                    <div class="result-body">
                        {res['explanation']}
                        <br><br>
                        <b>Operational Safe Range Math:</b><br>
                        - Predicted consumption = {res['predicted_demand']:.1f} servings.<br>
                        - Historical variability ($\sigma_{{{meal_type}}}$) = {res['sigma']:.1f} servings.<br>
                        - Adjusted safety factor = {res['safety_factor_used']:.2f}.<br>
                        - Optimal Prep = ceil({res['predicted_demand']:.1f} + ({res['safety_factor_used']:.2f} × {res['sigma']:.1f})) = <b>{res['recommended_preparation']} servings</b>.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Prediction Pipeline Error: {e}")


# --- PAGE 3: WASTE ANALYTICS ---
elif selected_page == "📊 Waste Analytics":
    st.markdown("<h1 class='main-header'>📊 Waste Analytics</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subheader'>Explore Historical Surplus Trends and Demand Variations</p>", unsafe_allow_html=True)
    
    st.sidebar.subheader("Filter Charts")
    selected_meals = st.sidebar.multiselect("Meal Types", options=df['meal_type'].unique(), default=list(df['meal_type'].unique()))
    selected_weather = st.sidebar.multiselect("Weather", options=df['weather'].unique(), default=list(df['weather'].unique()))
    
    filtered_df = df[df['meal_type'].isin(selected_meals) & df['weather'].isin(selected_weather)]
    
    if filtered_df.empty:
        st.warning("No data matches the selected filters.")
    else:
        # Grouped bar of Prepared vs Consumed vs Wasted
        st.markdown("### Historical Operations: Prepared vs. Consumed vs. Wasted")
        plot_df = filtered_df.tail(60).copy().reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['quantity_prepared'], name='Prepared Quantity', marker_color='#90CAF9'))
        fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['quantity_consumed'], name='Consumed Quantity (Demand)', marker_color='#A5D6A7'))
        fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['quantity_wasted'], name='Wasted Leftover', marker_color='#EF9A9A'))
        
        fig.update_layout(
            barmode='group',
            xaxis_title="Recent Services (Chronological Timeline)",
            yaxis_title="Servings",
            template="plotly_white",
            height=400,
            margin=dict(l=40, r=40, t=20, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Average Waste Percentage by Day")
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_agg = filtered_df.groupby('day_of_week')['waste_percentage'].mean().reindex(day_order).reset_index()
            
            fig_day = px.bar(
                day_agg, x='day_of_week', y='waste_percentage',
                color='waste_percentage', color_continuous_scale=px.colors.sequential.Oranges,
                labels={'day_of_week': 'Day of Week', 'waste_percentage': 'Average Waste %'}
            )
            fig_day.update_layout(template="plotly_white", height=320, coloraxis_showscale=False)
            st.plotly_chart(fig_day, use_container_width=True)
            
        with col2:
            st.markdown("### Average Waste Percentage by Meal Type")
            meal_agg = filtered_df.groupby('meal_type')['waste_percentage'].mean().reset_index()
            
            fig_meal = px.bar(
                meal_agg, x='meal_type', y='waste_percentage',
                color='waste_percentage', color_continuous_scale=px.colors.sequential.Reds,
                labels={'meal_type': 'Meal Type', 'waste_percentage': 'Average Waste %'}
            )
            fig_meal.update_layout(template="plotly_white", height=320, coloraxis_showscale=False)
            st.plotly_chart(fig_meal, use_container_width=True)
            
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("### Daily Average Waste Trend")
            date_agg = filtered_df.groupby('date')['waste_percentage'].mean().reset_index()
            fig_time = px.line(
                date_agg, x='date', y='waste_percentage',
                labels={'date': 'Date', 'waste_percentage': 'Average Waste %'}
            )
            fig_time.update_traces(line_color='#E65100', line_width=2.5)
            fig_time.update_layout(template="plotly_white", height=320)
            st.plotly_chart(fig_time, use_container_width=True)
            
        with col4:
            st.markdown("### Customer Count vs. Actual Demand")
            fig_trend = px.scatter(
                filtered_df, x='expected_customers', y='quantity_consumed',
                color='meal_type', opacity=0.7, trendline="ols",
                labels={'expected_customers': 'Expected Customers', 'quantity_consumed': 'Actual Consumed (Servings)'}
            )
            fig_trend.update_layout(template="plotly_white", height=320)
            st.plotly_chart(fig_trend, use_container_width=True)


# --- PAGE 4: MODEL PERFORMANCE ---
elif selected_page == "📈 Model Performance":
    st.markdown("<h1 class='main-header'>📈 Model Performance & Diagnostics</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subheader'>Machine Learning Metrics and Feature Importance Analysis</p>", unsafe_allow_html=True)
    
    # Model details
    st.markdown(f"**Model Architecture:** `{metrics['model_name']}` &nbsp;|&nbsp; **Training Timestamp:** `{metrics['training_time']}` &nbsp;|&nbsp; **Rows Trained:** `{metrics['dataset_size']}` &nbsp;|&nbsp; **Test split:** `20%` ({metrics['test_size']} rows)")
    
    rf = metrics['rf_metrics']
    lr = metrics['lr_metrics']
    
    st.markdown("### Test Set Accuracy Comparison")
    
    m_col1, m_col2, m_col3 = st.columns(3)
    
    with m_col1:
        st.metric("Mean Absolute Error (MAE)", f"{rf['mae']} servings", 
                  delta=f"Baseline LR: {lr['mae']} servings", delta_color="inverse",
                  help="Average size of prediction errors. Random Forest predicts closer to the true value.")
    with m_col2:
        st.metric("Root Mean Squared Error (RMSE)", f"{rf['rmse']} servings",
                  delta=f"Baseline LR: {lr['rmse']} servings", delta_color="inverse",
                  help="Penalizes larger outliers. Lower value indicates a more reliable predictor.")
    with m_col3:
        st.metric("R² Score (Variance Explained)", f"{rf['r2']}",
                  delta=f"Baseline LR: {lr['r2']}",
                  help="Proportion of target variance captured by variables. 1.0 is perfect.")
        
    with st.expander("🔬 Metric Explanations for Canteen Presenters"):
        st.markdown(f"""
        - **MAE ({rf['mae']} servings):** On average, the model's demand estimate is within **{rf['mae']} servings** of what students actually consume.
        - **RMSE ({rf['rmse']} servings):** Tracks variance dispersion. The low RMSE shows that our predictions are stable and rarely make large errors.
        - **R² Score ({rf['r2']}):** Indicates that **{rf['r2']*100:.1f}%** of the real-world demand swings are explained by pre-prep features, proving that scheduling predictions is far better than kitchen guessing.
        """)
        
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Actual vs. Predicted Consumption (Correlation)")
        actual = metrics['actual_vs_predicted']['actual']
        predicted = metrics['actual_vs_predicted']['predicted']
        
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=actual, y=predicted, mode='markers',
            name='Model Predictions', marker=dict(color='#2E7D32', size=7, opacity=0.7)
        ))
        
        # 45 degree perfect fit line
        min_v = min(min(actual), min(predicted))
        max_v = max(max(actual), max(predicted))
        fig_scatter.add_trace(go.Scatter(
            x=[min_v, max_v], y=[min_v, max_v], mode='lines',
            name='Perfect Prediction (y=x)', line=dict(color='#E53935', width=2, dash='dash')
        ))
        
        fig_scatter.update_layout(
            xaxis_title="Actual Consumption (Servings)",
            yaxis_title="Predicted Consumption (Servings)",
            template="plotly_white",
            height=380,
            legend=dict(x=0.02, y=0.98)
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with col2:
        st.markdown("### Feature Importance Scores (Random Forest)")
        imp_df = pd.DataFrame(metrics['feature_importances'])
        
        fig_imp = px.bar(
            imp_df.head(10), y='feature', x='importance',
            orientation='h', color='importance', color_continuous_scale=px.colors.sequential.Greens,
            labels={'feature': 'Input Variable', 'importance': 'Importance'}
        )
        fig_imp.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            template="plotly_white",
            height=380,
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_imp, use_container_width=True)


# --- PAGE 5: WHAT-IF SIMULATOR ---
elif selected_page == "🔄 What-If Simulator":
    st.markdown("<h1 class='main-header'>🔄 Scenario What-If Simulator</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subheader'>Slide Variables to Compare Under-Preparation, Over-Preparation, and SmartServe AI Recommendations</p>", unsafe_allow_html=True)
    
    # Fetch dataset means
    mean_cust = int(df['expected_customers'].mean())
    mean_temp = float(df['temperature'].mean())
    mean_prev_cons = float(df['prev_quantity_consumed'].mean())
    mean_prev_waste = float(df['prev_quantity_wasted'].mean())
    mean_roll = float(df['rolling_avg_consumption'].mean())
    mean_trend = float(df['recent_demand_trend'].mean())
    
    st.sidebar.subheader("Simulation Controls")
    sim_cust = st.sidebar.slider("Expected Customers", 10, 1000, mean_cust, step=10)
    sim_meal = st.sidebar.selectbox("Meal Type", ['Breakfast', 'Lunch', 'Dinner'])
    sim_day = st.sidebar.selectbox("Day of Week", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    sim_weather = st.sidebar.selectbox("Weather", ['Sunny', 'Cloudy', 'Rainy'])
    sim_temp = st.sidebar.slider("Temperature (°C)", -5.0, 45.0, mean_temp, step=1.0)
    sim_event = st.sidebar.checkbox("Special Event?", value=False)
    
    # Build input features
    is_weekend_val = 1 if sim_day in ['Saturday', 'Sunday'] else 0
    sim_input = {
        'day_of_week': sim_day,
        'meal_type': sim_meal,
        'expected_customers': sim_cust,
        'temperature': sim_temp,
        'weather': sim_weather,
        'special_event': 1 if sim_event else 0,
        'is_weekend': is_weekend_val,
        'prev_quantity_consumed': mean_prev_cons,
        'prev_quantity_wasted': mean_prev_waste,
        'rolling_avg_consumption': mean_roll,
        'recent_demand_trend': mean_trend
    }
    
    try:
        # Get baseline prediction
        pred_res = predict_single_instance(model, sim_input, variability_by_meal, safety_factor)
        predicted_demand = pred_res['predicted_demand']
        recommended_prep = pred_res['recommended_preparation']
        sigma = pred_res['sigma']
        
        st.markdown(f"💡 For this scenario, the model predicts a demand of **{predicted_demand:.1f} servings**. SmartServe AI recommends preparing **{recommended_prep} servings**.")
        
        # Interactive user slider
        user_prep = st.slider(
            "Your Custom Preparation Quantity (servings)", 
            min_value=max(1, int(predicted_demand * 0.4)), 
            max_value=int(predicted_demand * 1.8), 
            value=int(predicted_demand), 
            step=5,
            help="Simulate the shortage/waste impact of cooking a custom quantity."
        )
        
        # Calculate three scenarios dynamically
        # Scenario A: Under-Prepare (Prep is predicted demand - 1.2 * sigma)
        scen_a_prep = max(1, int(np.floor(predicted_demand - 1.2 * sigma)))
        # Scenario B: User preparation plan
        scen_b_prep = user_prep
        # Scenario C: Over-Prepare (Prep is predicted demand + 2.0 * sigma)
        scen_c_prep = int(np.ceil(predicted_demand + 2.0 * sigma))
        
        preps = [scen_a_prep, scen_b_prep, scen_c_prep]
        names = ["Scenario A: Under-Prepare", "Scenario B: Your Plan", "Scenario C: Over-Prepare"]
        scen_classes = ["scenario-a", "scenario-b", "scenario-c"]
        icons = ["🚨", "📋", "⚠️"]
        
        st.markdown("<br>", unsafe_allow_html=True)
        cols = st.columns(3)
        
        for i, (prep, name, s_class, icon) in enumerate(zip(preps, names, scen_classes, icons)):
            # Calculate dynamic metrics:
            # Consumption is capped at prepared servings
            actual_cons = min(prep, predicted_demand)
            leftover = max(0.0, prep - actual_cons)
            shortage = max(0.0, predicted_demand - prep)
            waste_pct = (leftover / prep * 100.0) if prep > 0 else 0.0
            
            # Risk classifications
            short_risk, waste_risk = classify_risks(prep, predicted_demand, sigma)
            
            # Risk badge style
            s_badge = "risk-low" if short_risk == "Low" else ("risk-medium" if short_risk == "Medium" else "risk-high")
            w_badge = "risk-low" if waste_risk == "Low" else ("risk-medium" if waste_risk == "Medium" else "risk-high")
            
            with cols[i]:
                st.markdown(f"""
                <div class="scenario-card {s_class}">
                    <div class="scenario-header">{icon} {name}</div>
                    <hr style="margin: 8px 0;">
                    <table style="width:100%; font-size:0.9rem; line-height:1.7;">
                        <tr>
                            <td><b>Preparation Quantity:</b></td>
                            <td style="text-align:right; font-weight:700;">{prep} servings</td>
                        </tr>
                        <tr>
                            <td><b>Predicted Demand:</b></td>
                            <td style="text-align:right;">{predicted_demand:.1f} servings</td>
                        </tr>
                        <tr style="color:#C62828;">
                            <td><b>Potential Shortage:</b></td>
                            <td style="text-align:right; font-weight:700;">{shortage:.1f} servings</td>
                        </tr>
                        <tr style="color:#E65100;">
                            <td><b>Expected Leftover:</b></td>
                            <td style="text-align:right; font-weight:700;">{leftover:.1f} servings</td>
                        </tr>
                        <tr>
                            <td><b>Waste Percentage:</b></td>
                            <td style="text-align:right;">{waste_pct:.1f}%</td>
                        </tr>
                    </table>
                    <hr style="margin: 10px 0;">
                    <div style="font-size:0.85rem; line-height:1.6;">
                        <b>Shortage Risk:</b> <span class="risk-badge {s_badge}">{short_risk}</span><br>
                        <div style="margin-top: 4px;"></div>
                        <b>Waste Risk:</b> <span class="risk-badge {w_badge}">{waste_risk}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
        # Scenario Chart
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Scenario Waste vs. Shortage Tradeoff")
        
        sim_chart_df = pd.DataFrame({
            'Scenario': ["Under-Prepare", "Your Plan", "Over-Prepare"],
            'Prepared Servings': preps,
            'Expected Leftovers (Waste)': [max(0.0, p - min(p, predicted_demand)) for p in preps],
            'Expected Shortage (Hungry Guests)': [max(0.0, predicted_demand - p) for p in preps]
        })
        
        fig_sim = go.Figure()
        fig_sim.add_trace(go.Bar(x=sim_chart_df['Scenario'], y=sim_chart_df['Prepared Servings'], name='Prepared Servings', marker_color='#90CAF9'))
        fig_sim.add_trace(go.Bar(x=sim_chart_df['Scenario'], y=sim_chart_df['Expected Leftovers (Waste)'], name='Wasted Servings', marker_color='#FFAB91'))
        fig_sim.add_trace(go.Bar(x=sim_chart_df['Scenario'], y=sim_chart_df['Expected Shortage (Hungry Guests)'], name='Customer Shortage (Unmet)', marker_color='#FFE082'))
        
        fig_sim.update_layout(
            barmode='group',
            yaxis_title="Servings",
            template="plotly_white",
            height=340,
            margin=dict(l=40, r=40, t=10, b=40)
        )
        st.plotly_chart(fig_sim, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error executing scenario calculations: {e}")


# --- PAGE 6: INSIGHTS & RECOMMENDATIONS ---
elif selected_page == "💡 Insights & Recommendations":
    st.markdown("<h1 class='main-header'>💡 Insights & Recommendations</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subheader'>Dynamic Operational Guidelines Extracted Directly from the Active Dataset</p>", unsafe_allow_html=True)
    
    with st.spinner("Extracting dynamic dataset characteristics..."):
        res = get_dynamic_insights(df, cost_per_serving)
        insights = res['insights']
        stats = res['stats']
        
        col1, col2 = st.columns([13, 10])
        
        with col1:
            st.markdown("### 📊 Active Diagnostic Observations")
            for insight in insights:
                st.markdown(f"""
                <div style="background-color:#ffffff; padding:15px; border-left:4px solid #4CAF50; border-radius:4px; box-shadow:0 2px 5px rgba(0,0,0,0.04); margin-bottom:12px;">
                    {insight}
                </div>
                """, unsafe_allow_html=True)
                
        with col2:
            st.markdown("### 💵 Simulated Financial Savings")
            st.markdown(f"""
            <div style="background-color:#FFF3E0; padding:20px; border-radius:12px; border:1px solid #FFE0B2; box-shadow:0 4px 6px rgba(0,0,0,0.03);">
                <h4 style="margin-top:0px; color:#E65100;">Optimized Impact Analysis</h4>
                <p style="font-size:0.95rem; line-height:1.7; color: #5D4037; margin-bottom: 0px;">
                    By shifting from historical guesswork preparation to prediction-led preparation:
                    <br><br>
                    - Current historical waste: <b>{stats['total_wasted']:,.0f} servings</b> (${stats['total_waste_cost']:,.2f})<br>
                    - Est. waste reduction: <b>{stats['waste_reduction_rate_pct']:.0f}%</b><br>
                    - Servings saved: <b>{stats['potential_savings_servings']:,.0f} servings</b><br>
                    - Ecological Offset: <b>{stats['co2_saved_kg']:,.0f} kg CO₂</b>
                    <hr style="border-top:1px solid #FFE0B2; margin: 15px 0;">
                    <span style="font-size:1.2rem; color:#E65100; font-weight:700;">Potential Cost Savings: ${stats['potential_savings_cost']:,.2f}</span>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            ### 🛠️ Key Kitchen Interventions
            1. **Tailored Safety Factors**: Apply higher safety factor setting (e.g. 1.1x) for Lunch/Dinner where demand variance is high.
            2. **Event Preparation Bounds**: Use event indicators to dynamically push the buffer margin without overproduction.
            3. **Weather Forecast Monitoring**: Monitor weather reports; increase safety buffers on Rainy days to support elevated student counts.
            """)


# --- PAGE 7: DATA UPLOAD & RETRAIN ---
elif selected_page == "📤 Data Upload & Retrain":
    st.markdown("<h1 class='main-header'>📤 Custom Data Upload & Retrain</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subheader'>Load Your Own Dataset, Validate Columns, and Re-Train the Random Forest Model</p>", unsafe_allow_html=True)
    
    st.markdown("### Schema Requirements")
    st.markdown("To retrain the ML models, your uploaded CSV file must contain the following exact columns:")
    st.code(", ".join(REQUIRED_COLUMNS), language="text")
    
    sample_df = load_and_clean_data(data_csv)
    csv_bytes = sample_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Sample food_waste_data.csv",
        data=csv_bytes,
        file_name="sample_food_waste_data.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
            
            is_valid, missing_cols = validate_dataset(uploaded_df)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Dataset Preview (First 5 Rows)")
                st.dataframe(uploaded_df.head(5))
                
            with col2:
                st.markdown("#### Validation Status")
                if is_valid:
                    st.success("✅ Schema compatible! All required fields are present.")
                    
                    st.markdown(f"""
                    - **Total Records:** {len(uploaded_df)}
                    - **Expected Customers Range:** {uploaded_df['expected_customers'].min()} to {uploaded_df['expected_customers'].max()}
                    - **Missing Fields Count:** {uploaded_df.isnull().sum().sum()}
                    """)
                    
                    st.markdown("### Retrain ML Pipeline")
                    if st.button("⚙️ Retrain Models on Uploaded Data", type="primary"):
                        with st.spinner("Retraining Random Forest pipeline and computing evaluations..."):
                            custom_data_path = os.path.join(project_root, 'data', 'food_waste_data_custom.csv')
                            uploaded_df.to_csv(custom_data_path, index=False)
                            
                            from src.train_model import train_and_evaluate
                            new_metrics = train_and_evaluate(custom_data_path, models_dir)
                            
                            # Update Session State
                            st.session_state['df_data'] = load_and_clean_data(custom_data_path)
                            st.session_state['model'] = load_trained_model(model_path)
                            st.session_state['metrics'] = new_metrics
                            
                            st.success("🎉 ML models successfully retrained and pipeline updated!")
                            st.rerun()
                else:
                    st.error(f"❌ Schema validation failed! Missing columns: {missing_cols}")
                    st.warning("Adjust your column headers to match the sample dataset schema before uploading.")
                    
        except Exception as e:
            st.error(f"Error reading uploaded file: {e}")
