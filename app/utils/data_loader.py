"""
Data loader utilities for DiabetesGuard Pro.
Author: Fahad
"""

import pandas as pd
import os
import sys

@pd.api.extensions.register_dataframe_accessor("diabetes")
class DiabetesAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
    
    @property
    def risk_factors(self):
        """Get the risk factor columns."""
        return [
            'Age', 'BMI', 'Blood_Pressure', 'Glucose_Level',
            'Exercise_Hours_Per_Week', 'Smoking_Status',
            'Alcohol_Consumption_Per_Week', 'Stress_Level'
        ]
    
    @property
    def categorical_columns(self):
        """Get categorical columns."""
        return ['Gender', 'Smoking_Status', 'Stress_Level']
    
    @property
    def numerical_columns(self):
        """Get numerical columns."""
        return [
            'Age', 'BMI', 'Blood_Pressure', 'Glucose_Level',
            'Exercise_Hours_Per_Week', 'Alcohol_Consumption_Per_Week'
        ]

def find_data_file():
    """Find the diabetes dataset in various possible locations."""
    # Get absolute paths
    current_file = os.path.abspath(__file__)
    app_utils_dir = os.path.dirname(current_file)
    app_dir = os.path.dirname(app_utils_dir)
    project_root = os.path.dirname(app_dir)
    
    # Possible locations for data file
    locations = [
        os.path.join(project_root, 'data', 'diabetes_dataset.csv'),  # /data directory
        os.path.join(project_root, 'diabetes_dataset.csv'),  # Root directory
    ]
    
    print("\nSearching for data file...")
    print(f"Current file: {current_file}")
    print(f"Project root: {project_root}")
    print(f"Possible locations: {locations}")
    
    # Find the data file
    for location in locations:
        if os.path.exists(location):
            print(f"Found data file at: {location}")
            return location
    
    # If we get here, we couldn't find the file
    raise FileNotFoundError(
        f"Could not find diabetes_dataset.csv in any of these locations: {locations}"
    )

def load_data():
    """Load the diabetes dataset."""
    try:
        print("\nStarting data loading...")
        data_path = find_data_file()
        
        print(f"Loading data from: {data_path}")
        df = pd.read_csv(data_path)
        print(f"Successfully loaded {len(df)} records")
        
        return df
        
    except Exception as e:
        print(f"\nERROR in load_data: {str(e)}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Python path: {sys.path}")
        raise

def get_feature_columns():
    """Get the list of feature columns used in the model."""
    # Get the absolute path to the models directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    feature_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))), 'models', 'feature_columns.txt')
    
    # Read and return feature columns
    with open(feature_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def get_feature_descriptions():
    """Get descriptions for each feature in the dataset."""
    return {
        'Age': 'Age in years',
        'Gender': 'Biological gender (Male/Female)',
        'BMI': 'Body Mass Index, a measure of body fat based on height and weight',
        'Blood_Pressure': 'Systolic blood pressure (mm Hg)',
        'Glucose_Level': 'Blood glucose level (mg/dL)',
        'Exercise_Hours_Per_Week': 'Hours of physical activity per week',
        'Smoking_Status': 'Current smoking status (Never/Former/Current)',
        'Alcohol_Consumption_Per_Week': 'Average alcoholic drinks per week',
        'Stress_Level': 'Self-reported stress level (Low/Moderate/High)',
        'Diabetes_Diagnosis': 'Whether the person has diabetes (0=No, 1=Yes)'
    }

def get_feature_ranges():
    """Get normal ranges for each numerical feature."""
    return {
        'Age': {'min': 18, 'max': 100, 'normal': '18-80'},
        'BMI': {'min': 18.5, 'max': 40, 'normal': '18.5-24.9'},
        'Blood_Pressure': {'min': 90, 'max': 180, 'normal': '90-120'},
        'Glucose_Level': {'min': 70, 'max': 200, 'normal': '70-100'},
        'Exercise_Hours_Per_Week': {'min': 0, 'max': 20, 'normal': '2.5-5'},
        'Alcohol_Consumption_Per_Week': {'min': 0, 'max': 21, 'normal': '0-7'}
    }

def get_risk_factors():
    """Get descriptions of major risk factors for diabetes."""
    return [
        {
            'factor': 'High Blood Glucose',
            'description': 'Fasting blood sugar > 126 mg/dL',
            'recommendation': 'Regular blood sugar monitoring and balanced diet'
        },
        {
            'factor': 'Obesity',
            'description': 'BMI > 30',
            'recommendation': 'Weight management through diet and exercise'
        },
        {
            'factor': 'Physical Inactivity',
            'description': 'Less than 150 minutes of exercise per week',
            'recommendation': 'Regular physical activity, aim for 30 minutes daily'
        },
        {
            'factor': 'High Blood Pressure',
            'description': 'Systolic BP > 140 mmHg',
            'recommendation': 'Blood pressure monitoring and lifestyle changes'
        },
        {
            'factor': 'Age',
            'description': 'Risk increases with age, especially after 45',
            'recommendation': 'Regular health check-ups and screenings'
        },
        {
            'factor': 'Smoking',
            'description': 'Current or former smoker',
            'recommendation': 'Smoking cessation and avoiding second-hand smoke'
        },
        {
            'factor': 'Alcohol Consumption',
            'description': 'More than 7 drinks per week',
            'recommendation': 'Limit alcohol intake and stay hydrated'
        },
        {
            'factor': 'Stress',
            'description': 'High levels of chronic stress',
            'recommendation': 'Stress management techniques and regular exercise'
        }
    ]
