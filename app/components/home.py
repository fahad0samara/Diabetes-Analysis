"""
Home component for DiabetesGuard Pro.
Author: Fahad
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import os
import sys

# Add parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from app.utils.data_loader import load_data

def render_home():
    """Render the home page."""
    st.title("🏥 Welcome to DiabetesGuard Pro")
    
    # Introduction
    st.markdown("""
    DiabetesGuard Pro is an advanced diabetes risk prediction tool that uses machine learning 
    to help you understand and manage your diabetes risk factors.
    """)
    
    # Key Features
    st.header("✨ Key Features")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - 🔮 **Advanced Risk Prediction**
        - 📊 **Interactive Data Visualization**
        - 💡 **Personalized Health Recommendations**
        - 📈 **Real-time Risk Assessment**
        """)
    
    with col2:
        st.markdown("""
        - 🎯 **Multiple Risk Factors Analysis**
        - 🧠 **Machine Learning Powered**
        - 📱 **User-Friendly Interface**
        - 🔒 **Privacy Focused**
        """)
    
    # Dataset Insights
    st.header("📊 Dataset Insights")
    df = load_data()
    
    # Display key statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_age = df['Age'].mean()
        st.metric("Average Age", f"{avg_age:.1f} years")
    
    with col2:
        avg_bmi = df['BMI'].mean()
        st.metric("Average BMI", f"{avg_bmi:.1f}")
    
    with col3:
        diabetes_rate = (df['Diabetes_Diagnosis'].mean() * 100)
        st.metric("Diabetes Rate", f"{diabetes_rate:.1f}%")
    
    # Visualizations
    st.header("📈 Data Visualizations")
    
    # Age distribution by diabetes status
    fig1 = px.histogram(df, x='Age', color='Diabetes_Diagnosis',
                       marginal='box', 
                       title='Age Distribution by Diabetes Status')
    st.plotly_chart(fig1)
    
    # BMI vs Glucose Level scatter plot
    fig2 = px.scatter(df, x='BMI', y='Glucose_Level',
                     color='Diabetes_Diagnosis',
                     title='BMI vs Glucose Level',
                     trendline="ols")
    st.plotly_chart(fig2)
    
    # Risk Factors
    st.header("⚠️ Key Risk Factors")
    st.markdown("""
    The following factors significantly influence diabetes risk:
    
    1. **High Blood Glucose**: Fasting blood sugar > 126 mg/dL
    2. **Elevated BMI**: BMI > 30 indicates obesity
    3. **High Blood Pressure**: Systolic BP > 140 mmHg
    4. **Physical Inactivity**: Less than 150 minutes/week
    5. **Family History**: Genetic predisposition
    6. **Age**: Risk increases with age
    7. **Unhealthy Diet**: High in sugar and processed foods
    8. **Stress Levels**: Chronic stress can affect blood sugar
    """)
    
    # Usage Instructions
    st.header("📝 How to Use")
    st.markdown("""
    1. Navigate to the **Predict** page
    2. Enter your health metrics:
        - Age, Gender, BMI
        - Blood Pressure, Glucose Level
        - Exercise Hours, Smoking Status
        - Alcohol Consumption, Stress Level
    3. Click "Predict Risk" to get your assessment
    4. Review your personalized recommendations
    
    **Note**: This tool is for informational purposes only and should not replace professional medical advice.
    """)
    
    render_footer()

def render_footer():
    """Render the footer section."""
    st.markdown("---")
    
    st.markdown("""
    <div class="creator-footer">
        Built with ❤️ by Fahad | © 2025 DiabetesGuard Pro
    </div>
    """, unsafe_allow_html=True)
