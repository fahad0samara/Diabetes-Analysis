"""
Model handler utilities for DiabetesGuard Pro.
Author: Fahad
"""

import joblib
import pandas as pd
import os
import sys

def find_model_files():
    """Find model files in various possible locations."""
    # Get absolute paths
    current_file = os.path.abspath(__file__)
    app_utils_dir = os.path.dirname(current_file)
    app_dir = os.path.dirname(app_utils_dir)
    project_root = os.path.dirname(app_dir)
    
    # Possible model directories
    model_dirs = [
        os.path.join(project_root, 'models'),  # Local development
        '/mount/src/diabetes-analysis/models',  # Streamlit Cloud
        os.path.join(os.getcwd(), 'models'),   # Alternative path
    ]
    
    print("\nSearching for model files...")
    print(f"Current file: {current_file}")
    print(f"Project root: {project_root}")
    print(f"Possible model directories: {model_dirs}")
    
    # Model filenames
    filenames = {
        'model': 'diabetes_model.joblib',
        'scaler': 'scaler.joblib',
        'features': 'feature_columns.txt'
    }
    
    # Find the first available model directory
    model_dir = None
    for dir_path in model_dirs:
        if os.path.exists(dir_path):
            print(f"Found models directory at: {dir_path}")
            model_dir = dir_path
            break
    
    if not model_dir:
        raise FileNotFoundError(f"Could not find models directory in any of: {model_dirs}")
    
    # Construct full file paths
    model_files = {
        file_type: os.path.join(model_dir, filename)
        for file_type, filename in filenames.items()
    }
    
    # Verify all files exist
    for file_type, file_path in model_files.items():
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Could not find {file_type} at: {file_path}")
        else:
            print(f"Found {file_type} at: {file_path}")
    
    return model_files

def load_model_components():
    """Load all model components."""
    try:
        print("\nStarting model component loading...")
        model_files = find_model_files()
        
        # Load components
        print("\nLoading model components...")
        model = joblib.load(model_files['model'])
        scaler = joblib.load(model_files['scaler'])
        
        # Load feature columns
        with open(model_files['features'], 'r') as f:
            feature_columns = [line.strip() for line in f.readlines()]
        
        print(f"Successfully loaded {len(feature_columns)} features")
        return model, scaler, feature_columns
        
    except Exception as e:
        print(f"\nERROR in load_model_components: {str(e)}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Python path: {sys.path}")
        print(f"Directory contents:")
        try:
            print("\nProject root contents:")
            print(os.listdir(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
            print("\nCurrent directory contents:")
            print(os.listdir(os.getcwd()))
        except Exception as list_err:
            print(f"Error listing directories: {str(list_err)}")
        raise

def prepare_input_data(input_data):
    """Prepare input data for prediction."""
    # Convert categorical variables
    mappings = {
        'Gender': {'Male': 0, 'Female': 1},
        'Smoking_Status': {'Never': 0, 'Former': 1, 'Current': 2},
        'Stress_Level': {'Low': 0, 'Moderate': 1, 'High': 2}
    }
    
    # Convert input_data to dictionary if it's a pandas Series
    if isinstance(input_data, pd.Series):
        input_data = input_data.to_dict()
    
    # Apply mappings to categorical variables
    for col, mapping in mappings.items():
        if col in input_data:
            if isinstance(input_data[col], str):
                input_data[col] = mapping.get(input_data[col], 0)
    
    return input_data

def get_prediction(input_data):
    """Get prediction for input data."""
    try:
        print("\nStarting prediction process...")
        model, scaler, feature_columns = load_model_components()
        
        print("Preparing input data...")
        input_data = prepare_input_data(input_data)
        input_df = pd.DataFrame([input_data])
        
        # Ensure all required features are present
        for col in feature_columns:
            if col not in input_df.columns:
                print(f"Adding missing column: {col}")
                input_df[col] = 0
        
        # Select and order features
        input_df = input_df[feature_columns]
        print(f"Input features: {input_df.columns.tolist()}")
        
        # Scale features
        print("Scaling features...")
        input_scaled = scaler.transform(input_df)
        
        # Get prediction and probability
        print("Making prediction...")
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        
        result = {
            'prediction': bool(prediction),
            'probability': float(probability),
            'risk_level': get_risk_level(probability)
        }
        print(f"Prediction result: {result}")
        return result
        
    except Exception as e:
        print(f"\nERROR in get_prediction: {str(e)}")
        raise

def get_risk_level(probability):
    """Get risk level based on probability."""
    if probability < 0.2:
        return "Low Risk"
    elif probability < 0.4:
        return "Moderate Risk"
    elif probability < 0.6:
        return "High Risk"
    else:
        return "Very High Risk"

def get_recommendations(input_data):
    """Get personalized health recommendations based on input values."""
    # Convert input_data to dictionary if it's a pandas Series
    if isinstance(input_data, pd.Series):
        input_data = input_data.to_dict()
    
    recommendations = []
    
    # BMI Recommendations
    bmi = input_data.get('BMI', 0)
    if bmi > 30:
        recommendations.append("🏋️ Weight Management:\n- Consider consulting a nutritionist\n- Aim for a balanced, calorie-controlled diet\n- Set realistic weight loss goals")
    elif bmi > 25:
        recommendations.append("⚖️ Weight Watch:\n- Monitor your caloric intake\n- Include more fruits and vegetables in your diet\n- Maintain regular physical activity")

    # Blood Pressure Recommendations
    bp = input_data.get('Blood_Pressure', 0)
    if bp > 140:
        recommendations.append("❤️ Blood Pressure Management:\n- Reduce sodium intake\n- Practice stress management techniques\n- Consider DASH diet\n- Regular BP monitoring")
    elif bp > 120:
        recommendations.append("🩺 Blood Pressure Watch:\n- Limit salt intake\n- Regular blood pressure monitoring\n- Stay physically active")

    # Glucose Level Recommendations
    glucose = input_data.get('Glucose_Level', 0)
    if glucose > 126:
        recommendations.append("🍎 Blood Sugar Control:\n- Monitor blood sugar regularly\n- Follow a balanced diet\n- Consider consulting an endocrinologist")
    elif glucose > 100:
        recommendations.append("🥗 Blood Sugar Watch:\n- Limit refined sugars\n- Choose whole grains over processed grains\n- Regular blood sugar monitoring")

    # Exercise Recommendations
    exercise = input_data.get('Exercise_Hours_Per_Week', 0)
    if exercise < 2.5:
        recommendations.append("🏃 Physical Activity:\n- Aim for at least 150 minutes of moderate exercise per week\n- Include both cardio and strength training\n- Start slowly and gradually increase intensity")

    # Smoking Recommendations
    smoking = input_data.get('Smoking_Status', '')
    if smoking == 'Current':
        recommendations.append("🚭 Smoking Cessation:\n- Consider nicotine replacement therapy\n- Join a smoking cessation program\n- Set a quit date\n- Seek support from family and friends")

    # Alcohol Recommendations
    alcohol = input_data.get('Alcohol_Consumption_Per_Week', 0)
    if alcohol > 14:
        recommendations.append("🍷 Alcohol Moderation:\n- Limit alcohol consumption\n- Stay within recommended guidelines\n- Consider alcohol-free days\n- Stay hydrated")

    # Stress Management
    stress = input_data.get('Stress_Level', '')
    if stress in ['High', 'Moderate']:
        recommendations.append("🧘 Stress Management:\n- Practice relaxation techniques\n- Consider meditation or yoga\n- Maintain a regular sleep schedule\n- Seek professional support if needed")

    # General Health Recommendations
    recommendations.append("🌟 General Health Tips:\n- Get regular health check-ups\n- Stay hydrated\n- Maintain a balanced diet\n- Get adequate sleep")
    
    return recommendations
