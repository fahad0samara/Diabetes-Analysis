"""
Model training script for DiabetesGuard Pro.
Author: Fahad
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

def create_advanced_features(df):
    """Create advanced features for the model."""
    # Create interaction features
    df['Age_BMI'] = df['Age'] * df['BMI']
    df['BMI_Glucose'] = df['BMI'] * df['Glucose_Level']
    
    # Create risk categories
    df['Age_Risk'] = pd.qcut(df['Age'], q=4, labels=[0, 1, 2, 3])
    df['BMI_Risk'] = pd.cut(df['BMI'], 
                           bins=[0, 18.5, 25, 30, float('inf')],
                           labels=[0, 1, 2, 3])
    df['BP_Risk'] = pd.cut(df['Blood_Pressure'],
                          bins=[0, 120, 140, float('inf')],
                          labels=[0, 1, 2])
    df['Glucose_Risk'] = pd.cut(df['Glucose_Level'],
                               bins=[0, 100, 126, float('inf')],
                               labels=[0, 1, 2])
    
    return df

def prepare_data():
    """Load and prepare the data for training."""
    # Get the absolute path to the data directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))), 'data', 'diabetes_dataset.csv')
    
    # Load data
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    # Create mappings for categorical variables
    mappings = {
        'Gender': {'Male': 0, 'Female': 1},
        'Smoking_Status': {'Never': 0, 'Former': 1, 'Current': 2},
        'Stress_Level': {'Low': 0, 'Moderate': 1, 'High': 2}
    }
    
    # Apply mappings
    for col, mapping in mappings.items():
        df[col] = df[col].map(mapping)
    
    # Create advanced features
    df = create_advanced_features(df)
    
    # Select features for the model
    feature_columns = [
        'Age', 'Gender', 'BMI', 'Blood_Pressure', 'Glucose_Level',
        'Exercise_Hours_Per_Week', 'Smoking_Status',
        'Alcohol_Consumption_Per_Week', 'Stress_Level',
        'Age_BMI', 'BMI_Glucose', 'Age_Risk', 'BMI_Risk',
        'BP_Risk', 'Glucose_Risk'
    ]
    
    X = df[feature_columns]
    y = df['Diabetes_Diagnosis']
    
    return X, y, feature_columns

def train_model():
    """Train the model and save all necessary files."""
    print("Loading and preparing data...")
    X, y, feature_columns = prepare_data()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize preprocessing objects
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    
    # Fit and transform the data
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    # Train the model
    print("Training model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    print("\nModel Performance:")
    print(classification_report(y_test, y_pred))
    
    # Get the absolute path to the models directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Save all necessary files
    print(f"\nSaving model files to: {models_dir}")
    joblib.dump(model, os.path.join(models_dir, 'diabetes_model.joblib'))
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.joblib'))
    joblib.dump(imputer, os.path.join(models_dir, 'imputer.joblib'))
    
    # Save feature columns
    with open(os.path.join(models_dir, 'feature_columns.txt'), 'w') as f:
        f.write('\n'.join(feature_columns))
    
    print("Training complete! Model files saved in the models directory.")

if __name__ == '__main__':
    train_model()
