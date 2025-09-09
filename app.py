import streamlit as st
import pandas as pd
import numpy as np

# --- PAGE CONFIGURATION ---
# This sets the title and icon of your app's browser tab.
st.set_page_config(
    page_title="AI Risk Prediction Engine",
    page_icon="🩺",
    layout="wide"
)

# --- DATA LOADING ---
# We use st.cache_data to load the data only once, making the app faster.
@st.cache_data
def load_data():
    # Load your final feature-engineered dataframe
    df = pd.read_parquet("final_df_for_app.parquet")
    
    # We need a unique ID for each patient for the cohort view.
    # Let's create one based on the index for this example.
    df['patient_id'] = [f'patient_{i//50}' for i in range(len(df))] # Example patient IDs
    
    return df

final_df = load_data()

# --- MODEL PREDICTIONS (Placeholder) ---
# In a real app, you would load your 'best_model.pkl' and make predictions.
# For this prototype, we'll create a placeholder "risk_score" column.
@st.cache_data
def get_predictions(df):
    summary_df = df.drop_duplicates(subset='patient_id', keep='last').copy()
    summary_df['risk_score'] = np.random.rand(len(summary_df)) * 100 
    return summary_df.sort_values(by='risk_score', ascending=False)

patient_summary_df = get_predictions(final_df)

# --- APP LAYOUT ---

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a View", ["Cohort View", "Patient Detail View"])

# --- COHORT VIEW PAGE ---
if page == "Cohort View":
    st.title("Patient Risk Cohort 🩺")
    st.markdown("This dashboard shows a list of all patients, sorted by their predicted 90-day deterioration risk.")
    
    # Display the main table of patients
    st.dataframe(
        patient_summary_df[['patient_id', 'age', 'risk_score']],
        use_container_width=True,
        height=600,
        column_config={
            "risk_score": st.column_config.ProgressColumn(
                "Risk Score",
                format="%.2f%%",
                min_value=0,
                max_value=100,
            ),
        }
    )

# --- PATIENT DETAIL VIEW PAGE ---
elif page == "Patient Detail View":
    st.title("Patient Detail and Risk Drivers 🧬")
    
    # Dropdown to select a patient
    selected_patient = st.selectbox(
        "Select a Patient ID to view details", 
        options=patient_summary_df['patient_id']
    )
    
    if selected_patient:
        # Get the data for the selected patient
        patient_data = patient_summary_df[patient_summary_df['patient_id'] == selected_patient].iloc[0]
        
        # Display key metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Patient ID", patient_data['patient_id'])
        col2.metric("Age", f"{patient_data['age']:.1f}")
        col3.metric("Risk Score", f"{patient_data['risk_score']:.2f}%", delta_color="inverse")
        
        st.markdown("---")
        
        # Display the SHAP plots and trend charts
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("Key Risk Drivers (Local Explanation)")
            st.image('shap_local_example.png')
            st.info("""
            **How to read this chart:**
            - **Red bars** show factors pushing this patient's risk score **higher**.
            - **Blue bars** show factors pulling the risk score **lower**.
            """)

        with col_right:
            st.subheader("Global Feature Importance")
            st.image('shap_global.png')

            st.subheader("Recommended Next Action")

            st.success("✅ **Recommendation:** Schedule a telehealth follow-up to discuss recent vital sign instability.")
