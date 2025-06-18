# In VS Code, save this as prediction_script.py
import joblib
import pandas as pd
import numpy as np

def create_BMI(df):
    return df['Weight'] / ((df['Height']/100) ** 2)

def create_Exercise_Intensity(df):
    return df['Duration'] * df['Heart_Rate']

def create_Thermal_Load(df):
    return df['Duration'] * df['Body_Temp']

def create_Work_Load(df):
    return df['Weight'] * df['Duration']

def create_Heart_Rate_Reserve(df):
    return df['Heart_Rate'] / (220 - df['Age'])

def create_MET_Approx(df):
    return (df['Heart_Rate'] - 60) / 10 + 1

def create_Energy_Score(df):
    return df['Duration'] * df['Heart_Rate'] * df['Body_Temp'] / 100

def apply_feature_engineering(df):
    """Apply all feature engineering transformations to the DataFrame"""
    df_engineered = df.copy()
    
    # Apply engineered features
    df_engineered['BMI'] = create_BMI(df_engineered)
    df_engineered['Exercise_Intensity'] = create_Exercise_Intensity(df_engineered)
    df_engineered['Thermal_Load'] = create_Thermal_Load(df_engineered)
    df_engineered['Work_Load'] = create_Work_Load(df_engineered)
    df_engineered['Heart_Rate_Reserve'] = create_Heart_Rate_Reserve(df_engineered)
    df_engineered['MET_Approx'] = create_MET_Approx(df_engineered)
    df_engineered['Energy_Score'] = create_Energy_Score(df_engineered)
    
    # Sex-based features
    df_engineered['Sex_Numeric'] = df_engineered['Sex'].map({'male': 1, 'female': 0})
    df_engineered['Sex_Weight'] = df_engineered['Sex_Numeric'] * df_engineered['Weight']
    df_engineered['Sex_Height'] = df_engineered['Sex_Numeric'] * df_engineered['Height']
    df_engineered['Sex_BMI'] = df_engineered['Sex_Numeric'] * df_engineered['BMI']
    
    return df_engineered

def predict_calories(new_data_path, submission_path=None):
    """
    Apply the ensemble model to new data
    
    Parameters:
    -----------
    new_data_path : str
        Path to the CSV file with new data (must have same columns as training data)
    submission_path : str, optional
        Path to save the Kaggle submission file
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with predicted calories
    """
    print(f"Loading models and data...")
    
    # Load the models
    rf_model = joblib.load('models/random_forest_optimized.pkl')
    xgb_model = joblib.load('models/xgboost_optimized.pkl')
    lgb_model = joblib.load('models/lightgbm_optimized.pkl')
    
    # Load the weights
    weights_df = pd.read_csv('models/ensemble_weights.csv')
    weights = weights_df['Weight'].tolist()
    
    # Load new data
    new_data = pd.read_csv(new_data_path)
    print(f"Loaded data with shape: {new_data.shape}")
    
    # Feature engineering
    print(f"Applying feature engineering...")
    df_engineered = apply_feature_engineering(new_data)
    
    # Prepare the features for prediction
    X = df_engineered.drop(['id'], axis=1, errors='ignore')
    if 'Calories' in X.columns:
        X = X.drop(['Calories'], axis=1)
    
    # Get dummy variables for categorical features
    X = pd.get_dummies(X, columns=['Sex'], drop_first=False)
    
    print(f"Prepared features with shape: {X.shape}")
    
    # Make predictions with each model
    print(f"Making predictions...")
    
    # Handle feature alignment for each model
    def align_features(X, model):
        """Align features to match the model's expected feature names"""
        model_features = model.feature_names_in_
        
        # Add missing columns
        for col in model_features:
            if col not in X.columns:
                X[col] = 0
                
        # Keep only the columns the model expects, in the right order
        return X[model_features]
    
    # Align features and predict for each model
    X_rf = align_features(X.copy(), rf_model)
    rf_pred = rf_model.predict(X_rf)
    
    X_xgb = align_features(X.copy(), xgb_model)
    xgb_pred = xgb_model.predict(X_xgb)
    
    X_lgb = align_features(X.copy(), lgb_model)
    lgb_pred = lgb_model.predict(X_lgb)
    
    # Weighted ensemble prediction
    print(f"Creating ensemble prediction with weights: {weights}")
    ensemble_pred = (
        weights[0] * rf_pred + 
        weights[1] * xgb_pred + 
        weights[2] * lgb_pred
    )
    
    # Create output dataframe
    predictions_df = pd.DataFrame({
        'id': new_data['id'],
        'Calories_RF': rf_pred,
        'Calories_XGB': xgb_pred,
        'Calories_LGB': lgb_pred,
        'Calories_Ensemble': ensemble_pred
    })
    
    print(f"Predictions complete.")
    
    # Create submission file if requested
    if submission_path:
        submission = predictions_df[['id', 'Calories_Ensemble']].rename(columns={'Calories_Ensemble': 'Calories'})
        submission.to_csv(submission_path, index=False)
        print(f"Submission file saved to {submission_path}")
    
    return predictions_df

if __name__ == "__main__":
    # Use command line arguments for more flexibility
    import argparse
    
    parser = argparse.ArgumentParser(description='Make predictions using the ensemble model')
    parser.add_argument('--test_data', type=str, default='playground-series-s5e5/test.csv', 
                        help='Path to test data CSV file')
    parser.add_argument('--output', type=str, default='predictions.csv',
                        help='Path to save the detailed predictions')
    parser.add_argument('--submission', type=str, default='submission.csv',
                        help='Path to save the Kaggle submission file')
    
    args = parser.parse_args()
    
    # Make predictions
    print(f"Processing file: {args.test_data}")
    predictions = predict_calories(args.test_data, args.submission)
    
    # Save detailed predictions
    predictions.to_csv(args.output, index=False)
    print(f"Detailed predictions saved to {args.output}")
    
    # Print summary statistics
    print("\nPrediction Summary:")
    print(f"Random Forest mean: {predictions['Calories_RF'].mean():.2f}")
    print(f"XGBoost mean: {predictions['Calories_XGB'].mean():.2f}")
    print(f"LightGBM mean: {predictions['Calories_LGB'].mean():.2f}")
    print(f"Ensemble mean: {predictions['Calories_Ensemble'].mean():.2f}")