"""
Predict component for DiabetesGuard Pro.
Author: Fahad
"""

import streamlit as st
import os
import sys

# Add parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from app.utils.model_handler import get_prediction, get_recommendations

def render_predict():
    """Render the prediction page."""
    st.title("🔮 Diabetes Risk Prediction")
    
    st.markdown("""
    Enter your health metrics below to get a personalized diabetes risk assessment 
    and health recommendations.
    """)
    
    # Create the prediction form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=0, max_value=120, value=30)
            gender = st.selectbox("Gender", ["Male", "Female"])
            bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
            blood_pressure = st.number_input("Blood Pressure (systolic)", min_value=70, max_value=200, value=120)
            glucose = st.number_input("Glucose Level (mg/dL)", min_value=70, max_value=300, value=100)
        
        with col2:
            exercise = st.number_input("Exercise Hours Per Week", min_value=0.0, max_value=40.0, value=3.0)
            smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
            alcohol = st.number_input("Alcohol Consumption (drinks per week)", min_value=0, max_value=50, value=0)
            stress = st.selectbox("Stress Level", ["Low", "Moderate", "High"])
        
        # Submit button
        submitted = st.form_submit_button("Predict Risk")
    
    # Make prediction when form is submitted
    if submitted:
        try:
            # Prepare input data
            input_data = {
                'Age': age,
                'Gender': gender,
                'BMI': bmi,
                'Blood_Pressure': blood_pressure,
                'Glucose_Level': glucose,
                'Exercise_Hours_Per_Week': exercise,
                'Smoking_Status': smoking,
                'Alcohol_Consumption_Per_Week': alcohol,
                'Stress_Level': stress
            }
            
            # Get prediction and recommendations
            result = get_prediction(input_data)
            recommendations = get_recommendations(input_data)
            
            # Display results
            st.markdown("## 📋 Risk Assessment")
            
            # Create columns for risk level and probability
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"### Risk Level: {result['risk_level']}")
                
            with col2:
                st.markdown(f"### Probability: {result['probability']:.1%}")
            
            # Add a progress bar for the probability
            st.progress(float(result['probability']))
            
            # Display recommendations
            st.markdown("## 💡 Health Recommendations")
            for recommendation in recommendations:
                st.markdown(recommendation)
                
        except Exception as e:
            st.error(f"An error occurred while making the prediction: {str(e)}")
            st.markdown("""
            Please try again. If the error persists, ensure that:
            1. All input values are within normal ranges
            2. The model files are properly loaded
            3. The system has sufficient resources
            """)

def render_footer():
    """Render the footer section."""
    st.markdown("---")
    st.markdown("⚕️ **Medical Disclaimer**: This tool provides general health recommendations based on your inputs. Always consult healthcare professionals for medical advice.")
    
    st.markdown("""
    <div class="creator-footer">
        Built with ❤️ by Fahad | &copy; 2025 DiabetesGuard Pro
    </div>
    """, unsafe_allow_html=True)
