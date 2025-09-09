import streamlit as st # type: ignore
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="AI Risk Prediction Engine",
    page_icon="🩺",
    layout="wide"
)

# to cache and load data faster
@st.cache_data
def load_data():
    # dataframe
    df = pd.read_csv("app_data.csv")
    
    # unique id for example
    df['patient_id'] = [f'patient_{i//50}' for i in range(len(df))] 
    
    return df

final_df = load_data()

# placeholder "risk_score" column
@st.cache_data
def get_predictions(df):
    summary_df = df.drop_duplicates(subset='patient_id', keep='last').copy()
    summary_df['risk_score'] = np.random.rand(len(summary_df)) * 100 
    return summary_df.sort_values(by='risk_score', ascending=False)

patient_summary_df = get_predictions(final_df)

# nav sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a View", ["Cohort View", "Patient Detail View"])

if page == "Cohort View":
    st.title("Patient Risk Cohort 🩺")
    st.markdown("This dashboard shows a list of all patients, sorted by their predicted 90-day deterioration risk.")
    
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

elif page == "Patient Detail View":
    st.title("Patient Detail and Risk Drivers 🧬")
    # dropdown
    selected_patient = st.selectbox(
        "Select a Patient ID to view details", 
        options=patient_summary_df['patient_id']
    )
    
    if selected_patient:
        patient_data = patient_summary_df[patient_summary_df['patient_id'] == selected_patient].iloc[0]
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Patient ID", patient_data['patient_id'])
        col2.metric("Age", f"{patient_data['age']:.1f}")
        col3.metric("Risk Score", f"{patient_data['risk_score']:.2f}%", delta_color="inverse")
        
        st.markdown("---")
        
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

