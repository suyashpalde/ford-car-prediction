"""
app.py
======
Car Price Intelligence: Industrial-Grade Vehicle Valuation Platform.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ==============================================================================
# 1. PAGE SETUP & GLOBAL LUXURY AUTOMOTIVE STYLING
# ==============================================================================
st.set_page_config(
    page_title="Car Price Intelligence | Automotive Valuation Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Design System: Dark Navy / Charcoal / Electric Cyan
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Core background */
    .stApp {
        background-color: #0B0F19;
        color: #F8FAFC;
    }

    /* Sidebar Customization */
    section[data-testid="stSidebar"] {
        background-color: #0F172A;
        border-right: 1px solid #1E293B;
    }

    /* Hero Section */
    .hero-container {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(11, 15, 25, 0.95) 100%),
                    url('https://images.unsplash.com/photo-1552519507-da3b142c6e3d?auto=format&fit=crop&w=1600&q=80') center/cover no-repeat;
        padding: 3rem 2.5rem;
        border-radius: 16px;
        border: 1px solid #1E293B;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5), 0 8px 10px -6px rgba(0, 0, 0, 0.5);
        margin-bottom: 2rem;
    }

    .hero-badge {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        background: rgba(6, 182, 212, 0.15);
        border: 1px solid #06B6D4;
        color: #06B6D4;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        border-radius: 9999px;
        margin-bottom: 1rem;
    }

    .hero-title {
        font-size: 2.75rem;
        font-weight: 800;
        color: #FFFFFF;
        line-height: 1.15;
        margin-bottom: 0.6rem;
        letter-spacing: -0.02em;
    }

    .hero-subtitle {
        font-size: 1.15rem;
        color: #94A3B8;
        max-width: 650px;
        line-height: 1.6;
        margin-bottom: 0;
    }

    /* Metric Cards */
    .metric-card {
        background-color: #131C2E;
        border: 1px solid #1E293B;
        padding: 1.5rem;
        border-radius: 14px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: #06B6D4;
        transform: translateY(-2px);
    }
    .metric-label {
        font-size: 0.825rem;
        font-weight: 600;
        text-transform: uppercase;
        color: #64748B;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    .metric-accent {
        color: #06B6D4;
    }

    /* Valuation Result Card */
    .result-card {
        background: radial-gradient(circle at top left, #1E293B 0%, #0F172A 100%);
        border: 1px solid #06B6D4;
        border-radius: 16px;
        padding: 2.25rem;
        text-align: center;
        box-shadow: 0 10px 30px -10px rgba(6, 182, 212, 0.3);
    }
    .result-price-gbp {
        font-size: 3.25rem;
        font-weight: 800;
        color: #38BDF8;
        letter-spacing: -0.03em;
        line-height: 1;
        margin: 1rem 0 0.5rem 0;
    }
    .result-price-inr {
        font-size: 1.25rem;
        font-weight: 500;
        color: #94A3B8;
    }

    /* Feature Specs Container */
    .spec-box {
        background-color: #131C2E;
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 1.25rem;
        margin-top: 1rem;
    }
    .spec-item {
        display: flex;
        justify-content: space-between;
        padding: 0.4rem 0;
        border-bottom: 1px solid #1E293B;
        font-size: 0.9rem;
    }
    .spec-item:last-child {
        border-bottom: none;
    }
    .spec-name {
        color: #64748B;
    }
    .spec-val {
        color: #E2E8F0;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. DATA & MODEL ARTIFACT LOADER
# ==============================================================================
@st.cache_resource
def load_assets():
    """Loads the dataset, trained pipeline, and benchmark metadata."""
    if not os.path.exists("artifacts/model_pipeline.joblib") or not os.path.exists("artifacts/metadata.json"):
        # Auto-train if artifacts haven't been exported yet
        from train_and_save import train_and_export
        train_and_export()

    pipeline = joblib.load("artifacts/model_pipeline.joblib")
    with open("artifacts/metadata.json", "r") as f:
        meta = json.load(f)

    # Load cleaned raw dataframe for explorer
    df = pd.read_csv("ford.csv")
    for col in ['model', 'transmission', 'fuelType']:
        df[col] = df[col].astype(str).str.strip()
    df = df[(df['year'] >= 1990) & (df['year'] <= 2024)]
    df = df[df['engineSize'] > 0.0].reset_index(drop=True)

    return pipeline, meta, df

try:
    pipeline, meta, df = load_assets()
except Exception as e:
    st.error(f"Initialization Error: {e}")
    st.info("Ensure `ford.csv` is in your project directory and run `python train_and_save.py`.")
    st.stop()


# ==============================================================================
# 3. SIDEBAR NAVIGATION & CONTROLS
# ==============================================================================
with st.sidebar:
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 1.5rem;">
            <div style="width: 38px; height: 38px; background: #06B6D4; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-weight: 800; color: #0B0F19; font-size: 1.2rem;">⚡</div>
            <div>
                <div style="font-weight: 700; font-size: 1rem; color: #FFFFFF; letter-spacing: -0.02em;">VALUATION AI</div>
                <div style="font-size: 0.75rem; color: #64748B;">Automotive Analytics</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["Home Dashboard", "Predict Car Price", "Model Insights", "Dataset Explorer", "About the Project"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Global Currency Converter Toggle
    st.markdown("<div class='metric-label'>Currency Settings</div>", unsafe_allow_html=True)
    show_inr = st.toggle("Show INR (₹) Conversion", value=True)
    exchange_rate = 106.50  # 1 GBP = ~106.50 INR
    if show_inr:
        exchange_rate = st.number_input("GBP to INR Rate", value=106.50, min_value=50.0, max_value=200.0, step=0.5)

    st.markdown("---")
    st.markdown("""
        <div style="font-size: 0.75rem; color: #475569; line-height: 1.4;">
            Verified Model: <b>RandomForest</b><br>
            Trained on <b>17,900+</b> Records<br>
            Dataset: <b>Ford UK Market</b>
        </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# PAGE 1: HOME DASHBOARD
# ==============================================================================
if page == "Home Dashboard":
    st.markdown("""
        <div class="hero-container">
            <span class="hero-badge">Next-Gen Intelligence</span>
            <h1 class="hero-title">Car Price Intelligence</h1>
            <p class="hero-subtitle">AI-powered predictive valuation engine for smarter automotive appraisal, real-time market benchmarking, and fleet portfolio analysis.</p>
        </div>
    """, unsafe_allow_html=True)

    # Key Stat Metrics Row
    stats = meta["dataset_stats"]
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)

    with m_col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Dataset Records</div>
                <div class="metric-value">{stats['total_records']:,}</div>
            </div>
        """, unsafe_allow_html=True)

    with m_col2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Average Listing Price</div>
                <div class="metric-value metric-accent">£{stats['mean_price']:,.0f}</div>
            </div>
        """, unsafe_allow_html=True)

    with m_col3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Model Precision (R²)</div>
                <div class="metric-value">{meta['metrics']['r2_score'] * 100:.1f}%</div>
            </div>
        """, unsafe_allow_html=True)

    with m_col4:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Mean Absolute Error</div>
                <div class="metric-value">±£{meta['metrics']['mae']:,.0f}</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Visualizations Row
    v_col1, v_col2 = st.columns([1.5, 1])

    with v_col1:
        st.markdown("### Market Price Distribution")
        fig_hist = px.histogram(
            df,
            x="price",
            nbins=60,
            color_discrete_sequence=['#06B6D4'],
            labels={'price': 'Price (£ GBP)'},
            template="plotly_dark"
        )
        fig_hist.update_layout(
            paper_bgcolor="#131C2E",
            plot_bgcolor="#131C2E",
            margin=dict(l=20, r=20, t=30, b=20),
            height=320,
            xaxis=dict(gridcolor="#1E293B"),
            yaxis=dict(gridcolor="#1E293B")
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with v_col2:
        st.markdown("### How the Valuation Works")
        st.markdown("""
        <div class="spec-box" style="line-height: 1.6; color: #94A3B8;">
            <p>Our valuation algorithm evaluates multi-dimensional vehicle attributes simultaneously:</p>
            <ul style="padding-left: 1.2rem; margin-bottom: 0.5rem;">
                <li><b style="color: #E2E8F0;">Non-linear Age & Mileage Decay:</b> Vehicles depreciate steeply during initial odometer milestones before leveling out.</li>
                <li><b style="color: #E2E8F0;">Powertrain & Efficiency Factors:</b> High MPG and lower road-tax brackets command market premiums.</li>
                <li><b style="color: #E2E8F0;">Variant Elasticity:</b> Premium models (Mustang, Edge) maintain independent valuation curves relative to volume trims.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# PAGE 2: PREDICT CAR PRICE
# ==============================================================================
elif page == "Predict Car Price":
    st.markdown("""
        <div style="margin-bottom: 1.5rem;">
            <h1 style="font-size: 2.2rem; font-weight: 800; color: #FFFFFF; margin-bottom: 0.3rem;">Predict Car Price</h1>
            <p style="color: #94A3B8; font-size: 1rem;">Configure the vehicle's parameters below for an instant machine-learning valuation.</p>
        </div>
    """, unsafe_allow_html=True)

    # Form Container
    with st.container():
        f_col1, f_col2, f_col3 = st.columns(3)

        with f_col1:
            st.markdown("#### Identification")
            car_model = st.selectbox("Vehicle Model", meta["features"]["models"], index=meta["features"]["models"].index("Fiesta") if "Fiesta" in meta["features"]["models"] else 0)
            year = st.slider("Registration Year", min_value=1998, max_value=2024, value=2018, step=1)
            transmission = st.selectbox("Transmission", meta["features"]["transmissions"], index=0)

        with f_col2:
            st.markdown("#### Usage & Mechanics")
            mileage = st.number_input("Odometer Reading (Miles)", min_value=0, max_value=200000, value=22000, step=1000)
            engine_size = st.slider("Engine Capacity (Litres)", min_value=1.0, max_value=5.0, value=1.0, step=0.1)
            fuel_type = st.selectbox("Fuel Type", meta["features"]["fuel_types"], index=0)

        with f_col3:
            st.markdown("#### Operational Economics")
            mpg = st.number_input("Combined Fuel Efficiency (MPG)", min_value=10.0, max_value=210.0, value=55.4, step=0.5)
            tax = st.number_input("Annual Road Tax (£)", min_value=0, max_value=600, value=145, step=5)

        st.markdown("<br>", unsafe_allow_html=True)
        col_btn1, col_btn2 = st.columns([1, 4])
        with col_btn1:
            predict_trigger = st.button("Calculate Valuation", type="primary", use_container_width=True)
        with col_btn2:
            preset = st.selectbox(
                "Or load typical preset:",
                ["-- Manual Configuration --", "Budget Commuter (Fiesta 2015)", "Family Compact (Focus 2018)", "Performance Sport (Mustang 2019)"],
                label_visibility="collapsed"
            )

    # Handle Presets
    if preset == "Budget Commuter (Fiesta 2015)":
        car_model, year, transmission, mileage, engine_size, fuel_type, mpg, tax = "Fiesta", 2015, "Manual", 45000, 1.2, "Petrol", 54.3, 30
        predict_trigger = True
    elif preset == "Family Compact (Focus 2018)":
        car_model, year, transmission, mileage, engine_size, fuel_type, mpg, tax = "Focus", 2018, "Manual", 18000, 1.5, "Diesel", 74.3, 145
        predict_trigger = True
    elif preset == "Performance Sport (Mustang 2019)":
        car_model, year, transmission, mileage, engine_size, fuel_type, mpg, tax = "Mustang", 2019, "Automatic", 8000, 5.0, "Petrol", 23.5, 145
        predict_trigger = True

    # Prediction Execution
    if predict_trigger:
        input_payload = pd.DataFrame([{
            'model': car_model,
            'year': int(year),
            'transmission': transmission,
            'mileage': int(mileage),
            'fuelType': fuel_type,
            'tax': int(tax),
            'mpg': float(mpg),
            'engineSize': float(engine_size)
        }])

        with st.spinner("Analyzing market comparables and running inference..."):
            predicted_gbp = float(pipeline.predict(input_payload)[0])
            predicted_gbp = max(500.0, predicted_gbp) # Business logic: floor at salvage value

        st.markdown("<br>", unsafe_allow_html=True)
        r_col1, r_col2 = st.columns([1.2, 1])

        with r_col1:
            inr_repr = f"₹ {(predicted_gbp * exchange_rate):,.0f} (approx. {(predicted_gbp * exchange_rate) / 100000:.2f} Lakhs)" if show_inr else ""
            st.markdown(f"""
                <div class="result-card">
                    <span class="hero-badge">AI MARKET APPRAISAL</span>
                    <div class="result-price-gbp">£{predicted_gbp:,.0f}</div>
                    <div class="result-price-inr">{inr_repr}</div>
                    <p style="color: #64748B; font-size: 0.8rem; margin-top: 1rem;">Model Estimate based on {meta['model_name']} with R² of {meta['metrics']['r2_score']}. Not an official financial guarantee.</p>
                </div>
            """, unsafe_allow_html=True)

        with r_col2:
            st.markdown(f"""
                <div class="spec-box">
                    <h4 style="color: #FFFFFF; margin-top: 0; margin-bottom: 0.75rem;">Vehicle Dossier</h4>
                    <div class="spec-item"><span class="spec-name">Model</span><span class="spec-val">Ford {car_model}</span></div>
                    <div class="spec-item"><span class="spec-name">Manufacturing Year</span><span class="spec-val">{year}</span></div>
                    <div class="spec-item"><span class="spec-name">Powertrain</span><span class="spec-val">{engine_size}L {fuel_type}</span></div>
                    <div class="spec-item"><span class="spec-name">Drivetrain</span><span class="spec-val">{transmission}</span></div>
                    <div class="spec-item"><span class="spec-name">Odometer</span><span class="spec-val">{mileage:,} Miles</span></div>
                    <div class="spec-item"><span class="spec-name">Fuel Economy</span><span class="spec-val">{mpg} MPG</span></div>
                    <div class="spec-item"><span class="spec-name">Annual Road Tax</span><span class="spec-val">£{tax}</span></div>
                </div>
            """, unsafe_allow_html=True)

        # Contextual Price Marker Graph
        st.markdown("### Valuation in Market Context")
        filtered_similar = df[df['model'] == car_model]['price']
        if len(filtered_similar) < 10:
            filtered_similar = df['price']

        fig_dist = px.histogram(
            filtered_similar,
            nbins=40,
            color_discrete_sequence=['#1E293B'],
            labels={'value': 'Price (£ GBP)'},
            template="plotly_dark",
            title=f"Price Distribution for Ford {car_model} across UK Market"
        )
        fig_dist.add_vline(x=predicted_gbp, line_width=3, line_dash="dash", line_color="#06B6D4", annotation_text="This Vehicle", annotation_font_color="#06B6D4")
        fig_dist.update_layout(
            paper_bgcolor="#131C2E",
            plot_bgcolor="#131C2E",
            height=280,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_dist, use_container_width=True)


# ==============================================================================
# PAGE 3: MODEL INSIGHTS
# ==============================================================================
elif page == "Model Insights":
    st.markdown("""
        <div style="margin-bottom: 1.5rem;">
            <h1 style="font-size: 2.2rem; font-weight: 800; color: #FFFFFF; margin-bottom: 0.3rem;">Model Performance & Diagnostics</h1>
            <p style="color: #94A3B8; font-size: 1rem;">Authentic evaluation metrics and residual diagnostics computed on an independent 20% test holdout.</p>
        </div>
    """, unsafe_allow_html=True)

    # Actual Metrics
    i_col1, i_col2, i_col3 = st.columns(3)
    m = meta['metrics']

    with i_col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Coefficient of Determination (R²)</div>
                <div class="metric-value metric-accent">{m['r2_score']}</div>
                <p style="color: #64748B; font-size: 0.8rem; margin-top: 0.5rem; margin-bottom: 0;">Explains {m['r2_score']*100:.1f}% of market variance.</p>
            </div>
        """, unsafe_allow_html=True)

    with i_col2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Mean Absolute Error (MAE)</div>
                <div class="metric-value">£{m['mae']:,.2f}</div>
                <p style="color: #64748B; font-size: 0.8rem; margin-top: 0.5rem; margin-bottom: 0;">Average dollar deviation per vehicle.</p>
            </div>
        """, unsafe_allow_html=True)

    with i_col3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Root Mean Squared Error (RMSE)</div>
                <div class="metric-value">£{m['rmse']:,.2f}</div>
                <p style="color: #64748B; font-size: 0.8rem; margin-top: 0.5rem; margin-bottom: 0;">Penalizes rare extreme valuation outliers.</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Test Set Scatter Plot (Actual vs. Predicted)
    st.markdown("### Test Set: Actual vs. Predicted Price")
    eval_df = pd.DataFrame(meta['eval_sample'])

    fig_eval = px.scatter(
        eval_df,
        x="Actual",
        y="Predicted",
        color_discrete_sequence=['#38BDF8'],
        opacity=0.65,
        template="plotly_dark",
        labels={'Actual': 'Actual Sale Price (£)', 'Predicted': 'Model Predicted Price (£)'}
    )
    # 45-degree reference line
    min_v = min(eval_df['Actual'].min(), eval_df['Predicted'].min())
    max_v = max(eval_df['Actual'].max(), eval_df['Predicted'].max())
    fig_eval.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode='lines', line=dict(color='#06B6D4', dash='dash'), name='Ideal Fit (y=x)'))

    fig_eval.update_layout(
        paper_bgcolor="#131C2E",
        plot_bgcolor="#131C2E",
        height=450,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(gridcolor="#1E293B"),
        yaxis=dict(gridcolor="#1E293B")
    )
    st.plotly_chart(fig_eval, use_container_width=True)

    st.markdown("### Metric Interpretations for Viva / Interview")
    st.markdown("""
    <div class="spec-box" style="color: #94A3B8; font-size: 0.9rem; line-height: 1.6;">
        <p><b style="color: #E2E8F0;">1. Why R² = ~0.90 matters:</b> Unlike a baseline Mean prediction model ($R^2=0$), our ensemble explains over 90% of price dispersion directly from odometer, engine displacement, and age features.</p>
        <p><b style="color: #E2E8F0;">2. Residual Analysis:</b> The scatter plot demonstrates strong adherence along the 45-degree ideal fit diagonal, showing symmetric error distribution across economy and executive categories.</p>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# PAGE 4: DATASET EXPLORER
# ==============================================================================
elif page == "Dataset Explorer":
    st.markdown("""
        <div style="margin-bottom: 1.5rem;">
            <h1 style="font-size: 2.2rem; font-weight: 800; color: #FFFFFF; margin-bottom: 0.3rem;">Automotive Market Explorer</h1>
            <p style="color: #94A3B8; font-size: 1rem;">Slice, filter, and inspect verified transactions from the UK Ford vehicle marketplace.</p>
        </div>
    """, unsafe_allow_html=True)

    # Interactive Filter Sidebar in Expander
    with st.expander("Filter Controls", expanded=True):
        c_filter1, c_filter2, c_filter3 = st.columns(3)
        with c_filter1:
            filter_models = st.multiselect("Select Models", meta["features"]["models"], default=["Fiesta", "Focus", "Kuga"])
        with c_filter2:
            filter_fuels = st.multiselect("Fuel Types", meta["features"]["fuel_types"], default=meta["features"]["fuel_types"])
        with c_filter3:
            price_bounds = st.slider("Price Filter (£)", int(df['price'].min()), int(df['price'].max()), (2000, 35000))

    # Apply Filters
    filtered_df = df[
        (df['model'].isin(filter_models if filter_models else meta["features"]["models"])) &
        (df['fuelType'].isin(filter_fuels if filter_fuels else meta["features"]["fuel_types"])) &
        (df['price'] >= price_bounds[0]) & (df['price'] <= price_bounds[1])
    ]

    st.markdown(f"Showing **{len(filtered_df):,}** matching records:")

    # Chart 1: Average Price by Model
    avg_price_model = filtered_df.groupby("model")['price'].mean().reset_index().sort_values(by="price", ascending=False)
    fig_bar = px.bar(
        avg_price_model,
        x="model",
        y="price",
        color="price",
        color_continuous_scale="Viridis",
        template="plotly_dark",
        labels={'price': 'Average Price (£)', 'model': 'Ford Model'},
        title="Average Asking Price by Model Trims"
    )
    fig_bar.update_layout(paper_bgcolor="#131C2E", plot_bgcolor="#131C2E", height=350, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_bar, use_container_width=True)

    # Searchable Raw Data Table
    st.markdown("### Searchable Transaction Records")
    st.dataframe(filtered_df, use_container_width=True, height=300)


# ==============================================================================
# PAGE 5: ABOUT THE PROJECT
# ==============================================================================
elif page == "About the Project":
    st.markdown("""
        <div style="margin-bottom: 1.5rem;">
            <h1 style="font-size: 2.2rem; font-weight: 800; color: #FFFFFF; margin-bottom: 0.3rem;">Project Architecture & Documentation</h1>
            <p style="color: #94A3B8; font-size: 1rem;">Complete technical breakdown for academic reviews and viva demonstrations.</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="spec-box" style="line-height: 1.7; color: #94A3B8;">
        <h3 style="color: #38BDF8; margin-top: 0;">1. Executive Summary</h3>
        <p>This project addresses the information asymmetry in secondary automotive markets. By engineering an end-to-end Machine Learning pipeline, it automates fair market valuation using historical vehicle sales data.</p>
        
        <h3 style="color: #38BDF8;">2. Pipeline & Feature Engineering</h3>
        <ul>
            <li><b>Categorical Encoding:</b> High-cardinality features (<code>model</code>, <code>transmission</code>, <code>fuelType</code>) are dynamically transformed via <code>OneHotEncoder</code> with <code>drop='first'</code> to avoid the dummy variable trap.</li>
            <li><b>Feature Standardization:</b> Continuous covariates (<code>year</code>, <code>mileage</code>, <code>tax</code>, <code>mpg</code>, <code>engineSize</code>) are scaled via <code>StandardScaler</code> ($\mu=0, \sigma=1$).</li>
            <li><b>Regression Algorithm:</b> Evaluated on both ordinary least squares (OLS) linear regressions and Random Forest tree ensembles. The Random Forest Regressor captures non-linear depreciation curvature.</li>
        </ul>

        <h3 style="color: #38BDF8;">3. College Viva Quick-Reference (Q&A)</h3>
        <ul>
            <li><b>Q: Why scale numerical variables if using tree-based algorithms?</b><br>
            A: While decision trees are scale-invariant, holding a standardized preprocessor inside a universal <code>ColumnTransformer</code> ensures model modularity (allowing easy swapping to Ridge, Lasso, or Neural Nets without rebuilding the data pipeline).</li>
            <li><b>Q: How are out-of-vocabulary categories handled?</b><br>
            A: The <code>OneHotEncoder</code> includes <code>handle_unknown='ignore'</code>, ensuring inference does not crash if an unseen trim is submitted.</li>
        </ul>

        <h3 style="color: #38BDF8;">4. Limitations and Future Scope</h3>
        <ul>
            <li><b>Geographic Dispersion:</b> Vehicle valuations in London/SE UK carry premiums not accounted for in base listings.</li>
            <li><b>Damage & Maintenance Records:</b> Future revisions could ingest MOT history API endpoints to evaluate salvage status.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)