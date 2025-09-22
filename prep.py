# prep.py - Data Preprocessing Module for Telco Churn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

def load_and_clean_data(filepath="data/WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    """Load and perform initial cleaning of the dataset"""
    df = pd.read_csv(filepath)
    
    # Convert TotalCharges to numeric, handling empty strings
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Fill missing TotalCharges with 0 (new customers)
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    # Remove customerID as it's not useful for prediction
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Missing values after cleaning: {df.isnull().sum().sum()}")
    
    return df

def encode_categorical_features(df):
    """Encode categorical variables"""
    df_encoded = df.copy()
    
    # Binary encoding for Yes/No columns
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    for col in binary_cols:
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].map({'Yes': 1, 'No': 0})
    
    # Gender encoding
    if 'gender' in df_encoded.columns:
        df_encoded['gender'] = df_encoded['gender'].map({'Male': 1, 'Female': 0})
    
    # MultipleLines, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
    service_cols = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
                   'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in service_cols:
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].map({'Yes': 2, 'No': 1, 'No phone service': 0, 'No internet service': 0})
    
    # One-hot encoding for multi-class categorical variables
    multi_class_cols = ['InternetService', 'Contract', 'PaymentMethod']
    df_encoded = pd.get_dummies(df_encoded, columns=multi_class_cols, drop_first=False)
    
    return df_encoded

def create_features(df):
    """Create additional features for better prediction"""
    df_feat = df.copy()
    
    # Average charge per month
    df_feat['AvgChargePerMonth'] = df_feat['TotalCharges'] / (df_feat['tenure'] + 1)
    
    # Charge difference from monthly
    df_feat['ChargeDifference'] = df_feat['MonthlyCharges'] - df_feat['AvgChargePerMonth']
    
    # Tenure groups
    df_feat['TenureGroup'] = pd.cut(df_feat['tenure'], 
                                    bins=[0, 12, 24, 48, 72], 
                                    labels=['0-1yr', '1-2yr', '2-4yr', '4-6yr'])
    df_feat = pd.get_dummies(df_feat, columns=['TenureGroup'], drop_first=False)
    
    # Service count (number of additional services)
    service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                   'TechSupport', 'StreamingTV', 'StreamingMovies']
    existing_service_cols = [col for col in service_cols if col in df_feat.columns]
    df_feat['ServiceCount'] = df_feat[existing_service_cols].apply(
        lambda x: sum(val == 2 for val in x), axis=1
    )
    
    # Has internet and phone
    if 'InternetService_No' in df_feat.columns:
        df_feat['HasBothServices'] = ((df_feat['PhoneService'] == 1) & 
                                      (df_feat['InternetService_No'] == 0)).astype(int)
    
    return df_feat

def split_data(df, target='Churn', test_size=0.2, val_size=0.1, random_state=42):
    """Split data into train, validation, and test sets"""
    
    # Separate features and target
    X = df.drop(target, axis=1)
    y = df[target]
    
    # First split: train+val and test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: train and validation
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
    )
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Target distribution:")
    print(f"  Train: {y_train.value_counts().to_dict()}")
    print(f"  Val: {y_val.value_counts().to_dict()}")
    print(f"  Test: {y_test.value_counts().to_dict()}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def scale_features(X_train, X_val, X_test, columns_to_scale=None):
    """Scale numerical features"""
    scaler = StandardScaler()
    
    if columns_to_scale is None:
        # Identify numerical columns
        columns_to_scale = X_train.select_dtypes(include=[np.number]).columns.tolist()
    
    # Fit on training data
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
    X_val_scaled[columns_to_scale] = scaler.transform(X_val[columns_to_scale])
    X_test_scaled[columns_to_scale] = scaler.transform(X_test[columns_to_scale])
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

def get_preprocessed_data(filepath=None, scale=True):
    """Complete preprocessing pipeline"""
    
    # Load and clean data
    if filepath is None:
        filepath = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = load_and_clean_data(filepath)
    
    # Encode categorical features
    df_encoded = encode_categorical_features(df)
    
    # Create additional features
    df_features = create_features(df_encoded)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_features)
    
    # Scale features if requested
    if scale:
        X_train, X_val, X_test, scaler = scale_features(X_train, X_val, X_test)
        return X_train, X_val, X_test, y_train, y_val, y_test, scaler
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test, output_dir=None):
    """Save processed data to files"""
    if output_dir is None:
        output_dir = "data/processed"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as CSV files
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_val.to_csv(os.path.join(output_dir, "X_val.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False, header=['Churn'])
    y_val.to_csv(os.path.join(output_dir, "y_val.csv"), index=False, header=['Churn'])
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False, header=['Churn'])
    
    print(f"Processed data saved to {output_dir}")

if __name__ == "__main__":
    # Run preprocessing pipeline
    print("Starting data preprocessing...")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = get_preprocessed_data()
    
    print("\nFeature information:")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Feature names: {list(X_train.columns[:10])}...")  # Show first 10
    
    # Save processed data
    save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test)
    print("\nPreprocessing complete!")
