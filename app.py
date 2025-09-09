import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# page config
st.set_page_config(
    page_title="AI Risk Prediction Engine",
    page_icon="🩺",
    layout="wide"
)

# load models and data
@st.cache_resource
def load_model_assets():
    model = joblib.load('best_model.joblib')
    explainer = joblib.load('shap_explainer.joblib')
    return model, explainer

@st.cache_data
def load_data():
    X_test = pd.read_pickle('X_test_data.pkl')
    y_test = pd.read_pickle('y_test_data.pkl')
    test_df = X_test.copy()
    test_df['DETERIORATION_TRUE'] = y_test
    return test_df

model, explainer = load_model_assets()
test_df = load_data()

# data prediction
@st.cache_data
def get_predictions():
    predictions = model.predict(test_df.drop('DETERIORATION_TRUE', axis=1))
    probabilities = model.predict_proba(test_df.drop('DETERIORATION_TRUE', axis=1))[:, 1]
    
    results_df = test_df.copy()
    results_df['risk_score'] = probabilities * 100
    results_df['DETERIORATION_PREDICTED'] = predictions
    results_df['patient_id'] = test_df.index # Assuming index is unique
    
    return results_df.sort_values(by='risk_score', ascending=False)

patient_summary_df = get_predictions()

# app layout
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a View", ["Cohort View", "Patient Detail View"])


# cohort view
if page == "Cohort View":
    st.title("Patient Risk Cohort 🩺")
    st.markdown("This dashboard shows a list of all patients, sorted by their predicted 90-day deterioration risk.")
    
    st.dataframe(
        patient_summary_df[['patient_id', 'age', 'risk_score']],
        use_container_width=True,
        height=600,
        column_config={
            "risk_score": st.column_config.ProgressColumn(
                "Risk Score", format="%.2f%%", min_value=0, max_value=100
            ),
        }
    )

# --- PATIENT DETAIL VIEW PAGE ---
elif page == "Patient Detail View":
    st.title("Patient Detail and Risk Drivers 🧬")
    
    selected_patient_index = st.selectbox(
        "Select a Patient to view details", 
        options=patient_summary_df.index
    )
    
    if selected_patient_index:
        patient_data = patient_summary_df.loc[selected_patient_index]
        patient_features = test_df.drop('DETERIORATION_TRUE', axis=1).loc[selected_patient_index]
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Patient Index", selected_patient_index)
        col2.metric("Age", f"{patient_data['age']:.1f}")
        col3.metric("Risk Score", f"{patient_data['risk_score']:.2f}%")
        
        st.markdown("---")
        
        st.subheader("Key Risk Drivers (Patient-Specific Explanation)")
        
        shap_values_single = explainer.shap_values(patient_features)
        
        fig, ax = plt.subplots()
        shap.force_plot(
            explainer.expected_value,
            shap_values_single,
            patient_features,
            matplotlib=True,
            show=False,
            text_rotation=15
        )
        st.pyplot(fig, bbox_inches='tight')
        
        st.info("""
        **How to read this chart:**
        - **Red bars** show factors pushing this patient's risk score **higher**.
        - **Blue bars** show factors pulling the risk score **lower**.
        """)
