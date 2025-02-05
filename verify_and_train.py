"""
Script to verify paths and train the diabetes prediction model.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def verify_paths():
    """Verify and create necessary directories."""
    # Get the project root directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define paths
    paths = {
        'data': os.path.join(current_dir, 'data'),
        'models': os.path.join(current_dir, 'models'),
        'dataset': os.path.join(current_dir, 'data', 'diabetes_dataset.csv')
    }
    
    # Create directories if they don't exist
    for path in ['data', 'models']:
        os.makedirs(paths[path], exist_ok=True)
        print(f"Verified {path} directory at: {paths[path]}")
    
    return paths

def load_and_preprocess_data(dataset_path):
    """Load and preprocess the dataset."""
    print("\nLoading dataset...")
    df = pd.read_csv(dataset_path)
    
    # Print dataset info
    print("\nDataset Info:")
    print(df.info())
    print("\nSample Data:")
    print(df.head())
    
    # Convert categorical variables
    categorical_columns = df.select_dtypes(include=['object']).columns
    label_encoders = {}
    
    print("\nProcessing categorical variables:")
    for column in categorical_columns:
        print(f"Encoding {column}...")
        label_encoders[column] = LabelEncoder()
        df[column] = label_encoders[column].fit_transform(df[column])
        print(f"Categories in {column}: {label_encoders[column].classes_}")
    
    # Separate features and target
    target_column = 'Diabetes_Diagnosis' if 'Diabetes_Diagnosis' in df.columns else 'Diabetes'
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # Save feature columns
    feature_columns = X.columns.tolist()
    print("\nFeature columns:", feature_columns)
    
    return X, y, feature_columns

def train_model(X, y):
    """Train the model."""
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training model...")
    model = DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    print("\nModel Performance:")
    print(f"Training Accuracy: {accuracy_score(y_train, y_train_pred):.2%}")
    print(f"Testing Accuracy: {accuracy_score(y_test, y_test_pred):.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    return model, scaler

def save_model_components(model, scaler, feature_columns, models_dir):
    """Save model components."""
    print("\nSaving model components...")
    
    # Save model
    model_path = os.path.join(models_dir, 'diabetes_model.joblib')
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    # Save scaler
    scaler_path = os.path.join(models_dir, 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to: {scaler_path}")
    
    # Save feature columns
    features_path = os.path.join(models_dir, 'feature_columns.txt')
    with open(features_path, 'w') as f:
        for column in feature_columns:
            f.write(f"{column}\n")
    print(f"Feature columns saved to: {features_path}")

def main():
    """Main function to verify paths and train model."""
    print("Starting model verification and training process...")
    
    # Verify paths
    paths = verify_paths()
    
    # Load and preprocess data
    X, y, feature_columns = load_and_preprocess_data(paths['dataset'])
    
    # Train model
    model, scaler = train_model(X, y)
    
    # Save components
    save_model_components(model, scaler, feature_columns, paths['models'])
    
    print("\nModel training and saving completed successfully!")

if __name__ == "__main__":
    main()
