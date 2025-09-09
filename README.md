# AI-Driven Risk Prediction Engine for Chronic Care

This project is a prototype of an AI-driven tool designed to predict the 90-day deterioration risk for patients with chronic conditions like diabetes.

**Live Demo:** [**https://smcechroniccare.streamlit.app/)**]

---

## Problem Statement

Predicting when a chronic care patient may deteriorate is a major clinical challenge. This tool aims to provide an early warning system for care teams, allowing for proactive intervention to improve patient outcomes and reduce hospitalizations.

## The Solution

This application uses a tuned XGBoost model trained on synthetic longitudinal patient data (vitals, diagnoses, encounters) to generate a risk score. The dashboard provides:
* A **Cohort View** to identify high-risk patients at a glance.
* A **Patient Detail View** that uses SHAP to explain the specific factors driving an individual's risk score.

## Key Features
* **Prediction Model:** Forecasts the probability of an ER visit or hospital admission in the next 90 days.
* **Explainability:** Uses SHAP to provide both global (overall) and local (patient-specific) explanations for its predictions.
* **Interactive Dashboard:** Built with Streamlit for a user-friendly clinical prototype.

## How to Run Locally
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`

---
