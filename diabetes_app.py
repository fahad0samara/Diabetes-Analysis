"""
DiabetesGuard Pro - Main Application
Author: Fahad
"""

import streamlit as st
import os
import sys
import joblib
import pandas as pd

# Add the app directory to Python path
app_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app')
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

# Import components
from components.home import render_home
from components.predict import render_predict
from components.analytics import render_analytics

def main():
    """Main application entry point."""
    # Set page config
    st.set_page_config(
        page_title="DiabetesGuard Pro",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Add CSS
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
        }
        .creator-footer {
            text-align: center;
            padding: 1rem;
            font-size: 0.8rem;
            color: #666;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Load the model and preprocessing objects
    @st.cache_resource
    def load_model():
        try:
            model = joblib.load('best_model.joblib')
            scaler = joblib.load('advanced_scaler.joblib')
            mappings = joblib.load('category_mappings.joblib')
            feature_columns = joblib.load('feature_columns.joblib')
            imputer = joblib.load('imputer.joblib')
            return model, scaler, mappings, feature_columns, imputer
        except FileNotFoundError:
            st.error("Model files not found. Please run the training script first.")
            return None, None, None, None, None
        except Exception as e:
            st.error(f"An error occurred while loading the model: {e}")
            return None, None, None, None, None

    model, scaler, mappings, feature_columns, imputer = load_model()

    # Add feature engineering functions after model loading
    def create_advanced_features(df):
        # Create interaction features
        df['Age_BMI'] = df['Age'] * df['BMI']
        df['BMI_Glucose'] = df['BMI'] * df['Glucose_Level']
        
        # Create risk categories
        df['Age_Risk'] = (df['Age'] > 50).astype(int)
        df['BMI_Risk'] = (df['BMI'] > 30).astype(int)
        df['BP_Risk'] = (df['Blood_Pressure'] > 140).astype(int)
        df['Glucose_Risk'] = (df['Glucose_Level'] > 126).astype(int)
        
        # Create exercise categories
        df['Exercise_Category'] = pd.cut(df['Exercise_Hours_Per_Week'], 
                                       bins=[0, 2, 4, float('inf')],
                                       labels=[0, 1, 2]).astype(int)
        
        # Stress level encoding (assuming it's already encoded in training)
        stress_map = {'Low': 0, 'Moderate': 1, 'High': 2}
        df['Stress_Level'] = df['Stress_Level'].map(stress_map)
        
        return df

    def get_recommendations(age, bmi, blood_pressure, glucose_level, exercise_hours, alcohol_consumption, smoking_status, stress_level):
        recommendations = []
        
        # BMI Recommendations
        if bmi > 30:
            recommendations.append("🏋️ Weight Management:\n- Consider consulting a nutritionist\n- Aim for a balanced, calorie-controlled diet\n- Set realistic weight loss goals")
        elif bmi > 25:
            recommendations.append("⚖️ Weight Watch:\n- Monitor your caloric intake\n- Include more fruits and vegetables in your diet\n- Maintain regular physical activity")

        # Blood Pressure Recommendations
        if blood_pressure > 140:
            recommendations.append("❤️ Blood Pressure Management:\n- Reduce sodium intake\n- Practice stress management techniques\n- Consider DASH diet\n- Regular BP monitoring")
        elif blood_pressure > 120:
            recommendations.append("🩺 Blood Pressure Watch:\n- Limit salt intake\n- Regular blood pressure monitoring\n- Stay physically active")

        # Glucose Level Recommendations
        if glucose_level > 126:
            recommendations.append("🍎 Blood Sugar Control:\n- Monitor blood sugar regularly\n- Consider consulting an endocrinologist\n- Follow a diabetes-friendly diet\n- Regular exercise routine")
        elif glucose_level > 100:
            recommendations.append("📊 Blood Sugar Watch:\n- Limit refined sugars and carbohydrates\n- Regular blood sugar monitoring\n- Include fiber-rich foods in your diet")

        # Exercise Recommendations
        if exercise_hours < 2.5:
            recommendations.append("🏃 Physical Activity:\n- Aim for at least 150 minutes of moderate exercise per week\n- Include both cardio and strength training\n- Start with walking and gradually increase intensity")

        # Alcohol Recommendations
        if alcohol_consumption > 7:
            recommendations.append("🍷 Alcohol Consumption:\n- Reduce alcohol intake\n- Limit to 1-2 drinks per day\n- Consider alcohol-free days")

        # Smoking Recommendations
        if smoking_status == "Current":
            recommendations.append("🚭 Smoking Cessation:\n- Consider nicotine replacement therapy\n- Join smoking cessation programs\n- Seek support from healthcare providers")

        # Stress Level Recommendations
        if stress_level == "High":
            recommendations.append("🧘 Stress Management:\n- Practice relaxation techniques\n- Consider meditation or yoga\n- Get adequate sleep\n- Consider counseling if needed")
        elif stress_level == "Moderate":
            recommendations.append("😌 Stress Reduction:\n- Regular exercise\n- Practice mindfulness\n- Maintain work-life balance")

        # General Recommendations
        recommendations.append("🏥 Regular Health Check-ups:\n- Annual physical examinations\n- Regular blood work\n- Eye examinations\n- Dental check-ups")
        
        return recommendations

    # Sidebar navigation
    st.sidebar.title("🏥 DiabetesGuard Pro")
    st.sidebar.markdown("---")
    
    # Navigation
    pages = {
        "🏠 Home": render_home,
        "🔮 Predict": render_predict,
        "📊 Analytics": render_analytics
    }
    
    page = st.sidebar.radio("Navigation", list(pages.keys()))
    
    # Render selected page
    pages[page]()
    
    # Sidebar footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **About DiabetesGuard Pro**
    
    A machine learning-powered diabetes risk assessment tool that helps you understand 
    and manage your diabetes risk factors.
    
    Built with ❤️ by Fahad
    """)

if __name__ == "__main__":
    main()
