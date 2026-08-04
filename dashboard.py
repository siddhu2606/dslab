import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier

# Set page configuration to wide layout to match the UI design
st.set_page_config(page_title="Alzheimer's Disease Analytics", layout="wide")

# Function to load default data if no file is uploaded
@st.cache_data
def load_default_data():
    try:
        return pd.read_csv('alzheimers_disease_data.csv')
    except FileNotFoundError:
        return None

# --- HEADER & CSV UPLOAD ---
st.title("NeuroTrack Clinical Analytics Dashboard")
st.markdown("Upload patient data to update the dashboard dynamically.")

uploaded_file = st.file_uploader("Upload Patient Data (CSV)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = load_default_data()

# Only render the dashboard if data is successfully loaded
if df is not None:
    # Set the target column for predictions/coloring
    target_col = 'Diagnosis' if 'Diagnosis' in df.columns else df.columns[-1]

    # --- KPI CARDS ---
    st.subheader("Overview")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    with kpi1:
        st.metric(label="Total Patients Monitored", value=f"{len(df):,}")
    with kpi2:
        avg_age = round(df['Age'].mean(), 1) if 'Age' in df.columns else "N/A"
        st.metric(label="Average Age", value=f"{avg_age} yrs")
    with kpi3:
        avg_chol = round(df['Cholesterol'].mean(), 1) if 'Cholesterol' in df.columns else "N/A"
        st.metric(label="Average Cholesterol", value=f"{avg_chol} mg/dL")
    with kpi4:
        high_risk = len(df[df[target_col] == 1]) if target_col in df.columns else "N/A"
        st.metric(label="High-Risk Alerts", value=high_risk, delta="Requires Review", delta_color="inverse")

    st.divider()

    # --- MAIN GRID LAYOUT ---
    col_main, col_side = st.columns([2, 1])

    with col_main:
        st.subheader("Patient Health Metric Analysis")
        
        # --- INTERACTIVE GRAPH CONTROLS ---
        ctrl1, ctrl2, ctrl3 = st.columns(3)
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        with ctrl1:
            x_axis = st.selectbox("X-Axis", options=numeric_cols, index=0)
        with ctrl2:
            y_axis = st.selectbox("Y-Axis", options=numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
        with ctrl3:
            graph_type = st.selectbox("Graph Type", options=["Scatter Plot", "Bar Chart", "Line Chart", "Density Heatmap"])

        # --- RENDER SELECTED GRAPH ---
        if graph_type == "Scatter Plot":
            fig = px.scatter(df, x=x_axis, y=y_axis, color=target_col, template="plotly_white")
        elif graph_type == "Bar Chart":
            fig = px.bar(df, x=x_axis, y=y_axis, color=target_col, template="plotly_white")
        elif graph_type == "Line Chart":
            df_sorted = df.sort_values(by=x_axis) # Sorting makes line charts readable
            fig = px.line(df_sorted, x=x_axis, y=y_axis, template="plotly_white")
        elif graph_type == "Density Heatmap":
            fig = px.density_heatmap(df, x=x_axis, y=y_axis, template="plotly_white")
        
        st.plotly_chart(fig, use_container_width=True)

        # --- PREDOMINANT FEATURES SECTION ---
        st.subheader("Predominant Risk Factors")
        if len(numeric_cols) > 1 and target_col in df.columns:
            # Train model to get feature importance
            X = df[numeric_cols].drop(columns=[target_col], errors='ignore')
            y = df[target_col]
            
            rf = RandomForestClassifier(random_state=42)
            rf.fit(X, y)
            
            # Format and sort feature importance
            importances = pd.DataFrame({'Feature': X.columns, 'Importance': rf.feature_importances_})
            importances = importances.sort_values(by='Importance', ascending=False).head(5)
            
            # Visualize top features
            fig_imp = px.bar(importances, x='Importance', y='Feature', orientation='h', template="plotly_white")
            fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_imp, use_container_width=True)

    with col_side:
        # --- PATIENT RISK PREDICTOR TOOL ---
        st.subheader("Patient Risk Predictor")
        with st.form("prediction_form"):
            st.write("Input metrics for real-time analysis")
            
            input_age = st.slider("Age", min_value=0, max_value=120, value=72)
            input_bmi = st.slider("BMI", min_value=10.0, max_value=50.0, value=26.5)
            input_sleep = st.slider("Sleep Quality (1-10)", min_value=1, max_value=10, value=6)
            input_chol = st.selectbox("Cholesterol Levels", ["Normal (<200 mg/dL)", "Borderline (200-239 mg/dL)", "High (>=240 mg/dL)"])
            
            submitted = st.form_submit_button("Predict Risk", type="primary")
            
            if submitted:
                # Placeholder logic for the prediction output
                st.info("Model processing... Ensure your trained model intercepts these slider variables to output the final prediction metric here.")

else:
    st.warning("Please upload a CSV file using the button above, or ensure 'alzheimers_disease_data.csv' is in the same directory as this script.")