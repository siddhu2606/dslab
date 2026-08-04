import streamlit as st

# ... [Your existing dashboard code above] ...

# --- EDUCATIONAL EXPLANATIONS SECTION ---
st.divider()
st.subheader("📚 Data Science Explainer Session")
st.write("Expand the sections below to understand the statistical concepts driving this dashboard.")

# 1. Graph Types Explainer
with st.expander("📊 Understanding the Graphs"):
    st.markdown("""
    **Visualizing Healthcare Data**
    Different graphs serve different purposes when analyzing patient metrics:
    *   **Scatter Plot:** Excellent for finding relationships between two continuous variables (e.g., checking if higher `Age` correlates with lower `Cognitive Score`). Each dot represents a single patient.
    *   **Bar Chart:** Best for comparing categorical data (e.g., comparing average health scores across different `Diagnosis` groups).
    *   **Line Chart:** Typically used for time-series data to spot trends over time, though it can also show progression across sorted sequential variables.
    *   **Density Heatmap:** Useful when you have thousands of data points that overlap on a scatter plot. It uses color intensity to show where patient data is most concentrated.
    """)

# 2. Skewness Explainer
with st.expander("📈 What is Skewness?"):
    st.markdown("""
    **Data Asymmetry**
    In a perfectly normal distribution (a bell curve), the data is symmetrical. **Skewness** measures how much the data leans to one side.
    
    *   **Positive Skew (Right-skewed):** The tail extends to the right. Most patients fall on the lower end of the scale, with a few extreme high values pulling the average up.
    *   **Negative Skew (Left-skewed):** The tail extends to the left. Most patients fall on the higher end of the scale (common in age distributions for diseases like Alzheimer's).
    
    *Statistical Formula for Skewness:*
    """)
    st.latex(r"Skewness = \frac{\sum_{i=1}^{N} (X_i - \bar{X})^3}{(N-1)s^3}")
    st.markdown("Where $\bar{X}$ is the mean and $s$ is the standard deviation.")

# 3. Outliers Explainer
with st.expander("🎯 Identifying Outliers"):
    st.markdown("""
    **Anomalies in Patient Data**
    An outlier is a data point that differs significantly from other observations. In healthcare, this could be a recording error, or a patient with exceptionally unique vitals (e.g., a highly unusual BMI or Cholesterol level).

    We commonly detect outliers using the **Interquartile Range (IQR)** method. The IQR is the middle 50% of the data:
    """)
    st.latex(r"IQR = Q_3 - Q_1")
    
    st.markdown("Any patient metric that falls below the Lower Bound or above the Upper Bound is flagged as an outlier:")
    st.latex(r"Lower\ Bound = Q_1 - (1.5 \times IQR)")
    st.latex(r"Upper\ Bound = Q_3 + (1.5 \times IQR)")

# 4. Feature Importance Explainer
with st.expander("🧠 Predominant Risk Factors (Feature Importance)"):
    st.markdown("""
    **How does the AI know what matters?**
    The bar chart on this dashboard uses a Machine Learning model (like a Random Forest) to calculate **Feature Importance**. 
    
    The model looks at all the variables (Age, BMI, Sleep Quality, etc.) and evaluates how much each metric helps in accurately predicting the `Diagnosis`. Variables with higher bars are the strongest predictors and carry the most weight in the patient's overall risk profile.
    """)

# ... [Rest of your code] ...
