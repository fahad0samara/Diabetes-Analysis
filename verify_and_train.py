"""
Verify and train diabetes prediction model.
Author: Fahad
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

def verify_and_train():
    """Verify files and train model if needed."""
    # Get absolute paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    models_dir = os.path.join(base_dir, 'models')
    
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Define paths
    data_path = os.path.join(data_dir, 'diabetes_dataset.csv')
    model_path = os.path.join(models_dir, 'diabetes_model.joblib')
    scaler_path = os.path.join(models_dir, 'scaler.joblib')
    features_path = os.path.join(models_dir, 'feature_columns.txt')
    
    print("\nVerifying paths:")
    print(f"Base directory: {base_dir}")
    print(f"Data directory: {data_dir}")
    print(f"Models directory: {models_dir}")
    print(f"Data path: {data_path}")
    print(f"Model path: {model_path}")
    print(f"Scaler path: {scaler_path}")
    print(f"Features path: {features_path}")
    
    # Verify data file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    print("\nLoading data...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} records")
    
    # Convert categorical variables
    print("\nProcessing features...")
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    df['Smoking_Status'] = df['Smoking_Status'].map({'Never': 0, 'Former': 1, 'Current': 2})
    df['Stress_Level'] = df['Stress_Level'].map({'Low': 0, 'Moderate': 1, 'High': 2})
    
    # Select features
    features = [
        'Age', 'Gender', 'BMI', 'Blood_Pressure', 'Glucose_Level',
        'Exercise_Hours_Per_Week', 'Smoking_Status',
        'Alcohol_Consumption_Per_Week', 'Stress_Level'
    ]
    
    # Save feature list
    print("\nSaving feature list...")
    with open(features_path, 'w') as f:
        f.write('\n'.join(features))
    print(f"Features saved to {features_path}")
    
    X = df[features]
    y = df['Diabetes_Diagnosis']
    
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\nTraining model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    print("\nEvaluating model...")
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    print(f"Training accuracy: {train_score:.4f}")
    print(f"Testing accuracy: {test_score:.4f}")
    
    print("\nSaving model files...")
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print("\nVerifying saved files...")
    if os.path.exists(model_path):
        print(f"Model saved successfully ({os.path.getsize(model_path)} bytes)")
    if os.path.exists(scaler_path):
        print(f"Scaler saved successfully ({os.path.getsize(scaler_path)} bytes)")
    if os.path.exists(features_path):
        print(f"Features saved successfully ({os.path.getsize(features_path)} bytes)")
    
    print("\nModel files saved successfully!")

if __name__ == "__main__":
    try:
        verify_and_train()
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise
