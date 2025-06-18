# In VS Code, save this as feature_engineering_ensemble.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Create directories for logs and plots
os.makedirs('logs', exist_ok=True)
os.makedirs('plots', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Set up logging
log_filename = f'logs/feature_engineering_ensemble_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# RMSLE (Root Mean Squared Logarithmic Error) - Competition Metric
def rmsle(y_true, y_pred):
    """
    Calculate Root Mean Squared Logarithmic Error
    RMSLE = sqrt(1/n * sum((log(1+y_true) - log(1+y_pred))^2))
    """
    # Ensure predictions are positive (since we're taking logs)
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(np.mean(np.power(np.log1p(y_true) - np.log1p(y_pred), 2)))

# Create a scorer for cross-validation and GridSearch
rmsle_scorer = make_scorer(rmsle, greater_is_better=False)

try:
    logger.info("Starting feature engineering and ensemble modeling...")
    
    # Load the data
    logger.info("Loading data...")
    try:
        data = pd.read_csv('playground-series-s5e5/train.csv')
        logger.info("Successfully loaded data from 'playground-series-s5e5/train.csv'")
    except FileNotFoundError:
        try:
            data = pd.read_csv('train.csv')
            logger.info("Successfully loaded data from 'train.csv'")
        except FileNotFoundError:
            logger.error("Could not find the training data file. Please ensure it's in the correct location.")
            raise
    
    # Display basic info
    logger.info(f"Dataset shape: {data.shape}")
    logger.info(f"Dataset columns: {data.columns.tolist()}")
    
    # Check for missing values
    missing_values = data.isnull().sum()
    if missing_values.sum() > 0:
        logger.warning(f"Dataset contains {missing_values.sum()} missing values. Handling missing values...")
        data = data.fillna(data.median())
    
    # ===============================================================
    # FEATURE ENGINEERING 
    # ===============================================================
    logger.info("Performing feature engineering...")
    
    # Create a copy of the original data for feature engineering
    df_engineered = data.copy()
    
    # 1. BMI (Body Mass Index)
    df_engineered['BMI'] = df_engineered['Weight'] / ((df_engineered['Height']/100) ** 2)
    logger.info("Created BMI feature")
    
    # 2. Exercise Intensity (Duration * Heart_Rate)
    df_engineered['Exercise_Intensity'] = df_engineered['Duration'] * df_engineered['Heart_Rate']
    logger.info("Created Exercise_Intensity feature")
    
    # 3. Thermal_Load (Duration * Body_Temp)
    df_engineered['Thermal_Load'] = df_engineered['Duration'] * df_engineered['Body_Temp']
    logger.info("Created Thermal_Load feature")
    
    # 4. Work_Load (Weight * Duration)
    df_engineered['Work_Load'] = df_engineered['Weight'] * df_engineered['Duration']
    logger.info("Created Work_Load feature")
    
    # 5. Age categories
    bins = [0, 30, 50, 100]
    labels = ['Young', 'Middle_Aged', 'Senior']
    df_engineered['Age_Group'] = pd.cut(df_engineered['Age'], bins=bins, labels=labels)
    logger.info("Created Age_Group feature")
    
    # 6. BMI categories
    bmi_bins = [0, 18.5, 25, 30, 100]
    bmi_labels = ['Underweight', 'Normal', 'Overweight', 'Obese']
    df_engineered['BMI_Category'] = pd.cut(df_engineered['BMI'], bins=bmi_bins, labels=bmi_labels)
    logger.info("Created BMI_Category feature")
    
    # 7. Heart Rate Reserve (HRR) = Heart_Rate / (220 - Age) (percentage of max heart rate)
    df_engineered['Heart_Rate_Reserve'] = df_engineered['Heart_Rate'] / (220 - df_engineered['Age'])
    logger.info("Created Heart_Rate_Reserve feature")
    
    # 8. Sex-based interactions
    # First encode Sex as a numeric value
    df_engineered['Sex_Numeric'] = df_engineered['Sex'].map({'male': 1, 'female': 0})
    
    # Then create interaction features
    df_engineered['Sex_Weight'] = df_engineered['Sex_Numeric'] * df_engineered['Weight']
    df_engineered['Sex_Height'] = df_engineered['Sex_Numeric'] * df_engineered['Height']
    df_engineered['Sex_BMI'] = df_engineered['Sex_Numeric'] * df_engineered['BMI']
    
    logger.info("Created Sex-based interaction features")
    
    # 9. Metabolic Equivalent of Task (MET) approximation
    # This is a domain-specific feature estimation
    df_engineered['MET_Approx'] = (df_engineered['Heart_Rate'] - 60) / 10 + 1
    logger.info("Created MET_Approx feature")
    
    # 10. Energy Expenditure Score
    df_engineered['Energy_Score'] = (
        df_engineered['Duration'] * 
        df_engineered['Heart_Rate'] * 
        df_engineered['Body_Temp'] / 
        100
    )
    logger.info("Created Energy_Score feature")
    
    # Log the new features
    logger.info(f"New engineered features: {[col for col in df_engineered.columns if col not in data.columns]}")
    
    # Prepare features and target for modeling
    # Drop unnecessary columns and categorical columns that need to be encoded
    X = df_engineered.drop(['id', 'Calories', 'Age_Group', 'BMI_Category'], axis=1)
    
    # Get dummy variables for categorical features
    X = pd.get_dummies(X, columns=['Sex'], drop_first=False)
    y = df_engineered['Calories']
    
    # Check for any remaining categorical columns
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        logger.info(f"Remaining categorical columns to be encoded: {cat_cols}")
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    
    logger.info(f"Final feature set shape: {X.shape}")
    logger.info(f"Feature list: {X.columns.tolist()}")
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info(f"Training set size: {X_train.shape}, Test set size: {X_test.shape}")
    
    # ===============================================================
    # HYPERPARAMETER TUNING 
    # ===============================================================
    logger.info("Starting hyperparameter tuning...")
    
    # Since we have a large dataset, we'll use RandomizedSearchCV with a reasonable number of iterations
    
    # 1. Random Forest Hyperparameter Tuning
    logger.info("Tuning Random Forest...")
    
    # Define the parameter grid for RandomForest
    rf_param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    
    # Create a RandomForestRegressor
    rf = RandomForestRegressor(random_state=42)
    
    # Use RandomizedSearchCV for more efficient searching
    rf_random = RandomizedSearchCV(
        estimator=rf,
        param_distributions=rf_param_dist,
        n_iter=20,  # Number of parameter settings sampled
        scoring=rmsle_scorer,
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    # Fit the randomized search
    rf_random.fit(X_train, y_train)
    
    # Get the best parameters and score
    rf_best_params = rf_random.best_params_
    rf_best_score = -rf_random.best_score_  # Convert back to positive as our scorer is negative
    
    logger.info(f"Random Forest best parameters: {rf_best_params}")
    logger.info(f"Random Forest best RMSLE score: {rf_best_score:.6f}")
    
    # Create the best Random Forest model
    best_rf = RandomForestRegressor(**rf_best_params, random_state=42)
    best_rf.fit(X_train, y_train)
    rf_pred = best_rf.predict(X_test)
    rf_rmsle = rmsle(y_test, rf_pred)
    logger.info(f"Optimized Random Forest RMSLE on test set: {rf_rmsle:.6f}")
    
    # 2. XGBoost Hyperparameter Tuning
    logger.info("Tuning XGBoost...")
    
    # Define the parameter grid for XGBoost
    xgb_param_dist = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7, 9],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'subsample': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.2],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [0, 0.1, 1],
        'min_child_weight': [1, 3, 5]
    }
    
    # Create an XGBoost regressor
    xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    
    # Use RandomizedSearchCV for more efficient searching
    xgb_random = RandomizedSearchCV(
        estimator=xgb_reg,
        param_distributions=xgb_param_dist,
        n_iter=20,  # Number of parameter settings sampled
        scoring=rmsle_scorer,
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    # Fit the randomized search
    xgb_random.fit(X_train, y_train)
    
    # Get the best parameters and score
    xgb_best_params = xgb_random.best_params_
    xgb_best_score = -xgb_random.best_score_  # Convert back to positive as our scorer is negative
    
    logger.info(f"XGBoost best parameters: {xgb_best_params}")
    logger.info(f"XGBoost best RMSLE score: {xgb_best_score:.6f}")
    
    # Create the best XGBoost model
    best_xgb = xgb.XGBRegressor(**xgb_best_params, objective='reg:squarederror', random_state=42)
    best_xgb.fit(X_train, y_train)
    xgb_pred = best_xgb.predict(X_test)
    xgb_rmsle = rmsle(y_test, xgb_pred)
    logger.info(f"Optimized XGBoost RMSLE on test set: {xgb_rmsle:.6f}")
    
    # 3. LightGBM Hyperparameter Tuning
    logger.info("Tuning LightGBM...")
    
    # Define the parameter grid for LightGBM
    lgb_param_dist = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [31, 50, 70],
        'max_depth': [-1, 10, 15, 20],
        'min_child_samples': [20, 30, 50],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [0, 0.1, 0.5]
    }
    
    # Create a LightGBM regressor
    lgb_reg = lgb.LGBMRegressor(objective='regression', random_state=42)
    
    # Use RandomizedSearchCV for more efficient searching
    lgb_random = RandomizedSearchCV(
        estimator=lgb_reg,
        param_distributions=lgb_param_dist,
        n_iter=20,  # Number of parameter settings sampled
        scoring=rmsle_scorer,
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    # Fit the randomized search
    lgb_random.fit(X_train, y_train)
    
    # Get the best parameters and score
    lgb_best_params = lgb_random.best_params_
    lgb_best_score = -lgb_random.best_score_  # Convert back to positive as our scorer is negative
    
    logger.info(f"LightGBM best parameters: {lgb_best_params}")
    logger.info(f"LightGBM best RMSLE score: {lgb_best_score:.6f}")
    
    # Create the best LightGBM model
    best_lgb = lgb.LGBMRegressor(**lgb_best_params, objective='regression', random_state=42)
    best_lgb.fit(X_train, y_train)
    lgb_pred = best_lgb.predict(X_test)
    lgb_rmsle = rmsle(y_test, lgb_pred)
    logger.info(f"Optimized LightGBM RMSLE on test set: {lgb_rmsle:.6f}")
    
    # ===============================================================
    # ENSEMBLE MODELING
    # ===============================================================
    logger.info("Creating an ensemble model with the optimized models...")
    
    # Calculate weights based on inverse of RMSLE (lower RMSLE = higher weight)
    # Sum of weights should be 1
    total_error = rf_rmsle + xgb_rmsle + lgb_rmsle
    weights = [
        1/rf_rmsle / (1/rf_rmsle + 1/xgb_rmsle + 1/lgb_rmsle),
        1/xgb_rmsle / (1/rf_rmsle + 1/xgb_rmsle + 1/lgb_rmsle),
        1/lgb_rmsle / (1/rf_rmsle + 1/xgb_rmsle + 1/lgb_rmsle)
    ]
    
    logger.info(f"Ensemble weights: RF={weights[0]:.4f}, XGB={weights[1]:.4f}, LGB={weights[2]:.4f}")
    
    # Create the weighted ensemble prediction
    ensemble_pred = (
        weights[0] * rf_pred + 
        weights[1] * xgb_pred + 
        weights[2] * lgb_pred
    )
    
    # Calculate and log ensemble performance
    ensemble_rmsle = rmsle(y_test, ensemble_pred)
    ensemble_mse = mean_squared_error(y_test, ensemble_pred)
    ensemble_r2 = r2_score(y_test, ensemble_pred)
    
    logger.info("Ensemble model performance:")
    logger.info(f"RMSLE (competition metric): {ensemble_rmsle:.6f}")
    logger.info(f"MSE: {ensemble_mse:.6f}")
    logger.info(f"R² Score: {ensemble_r2:.6f}")
    
    # Compare individual models with ensemble
    results_df = pd.DataFrame({
        'Model': ['Random Forest', 'XGBoost', 'LightGBM', 'Ensemble'],
        'RMSLE': [rf_rmsle, xgb_rmsle, lgb_rmsle, ensemble_rmsle],
        'MSE': [mean_squared_error(y_test, rf_pred), 
                mean_squared_error(y_test, xgb_pred), 
                mean_squared_error(y_test, lgb_pred), 
                ensemble_mse],
        'R²': [r2_score(y_test, rf_pred), 
               r2_score(y_test, xgb_pred), 
               r2_score(y_test, lgb_pred), 
               ensemble_r2],
        'Weight': [weights[0], weights[1], weights[2], 'N/A']
    })
    
    logger.info("\nFinal Model Comparison:")
    logger.info(f"\n{results_df.to_string(index=False)}")
    
    # Save the results to a CSV file
    results_df.to_csv('model_comparison_results_ensemble.csv', index=False)
    
    # Create a bar chart comparing the models
    plt.figure(figsize=(10, 6))
    plt.bar(results_df['Model'], results_df['RMSLE'], color='skyblue')
    plt.xlabel('Models')
    plt.ylabel('RMSLE (Lower is Better)')
    plt.title('Model Comparison by RMSLE')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add RMSLE values on top of each bar
    for i, v in enumerate(results_df['RMSLE']):
        plt.text(i, v + 0.001, f'{v:.6f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('plots/ensemble_comparison.png')
    
    # Feature importance analysis
    logger.info("Analyzing feature importance...")
    
    # Get feature importance from Random Forest
    rf_feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Log top features
    logger.info("Top 10 important features from Random Forest:")
    logger.info(f"\n{rf_feature_importance.head(10).to_string(index=False)}")
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=rf_feature_importance.head(15))
    plt.title('Top 15 Features by Importance (Random Forest)')
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png')
    
    # Save the best models for later use
    import joblib
    joblib.dump(best_rf, 'models/random_forest_optimized.pkl')
    joblib.dump(best_xgb, 'models/xgboost_optimized.pkl')
    joblib.dump(best_lgb, 'models/lightgbm_optimized.pkl')
    
    # Store weights
    weights_df = pd.DataFrame({
        'Model': ['RandomForest', 'XGBoost', 'LightGBM'],
        'Weight': weights
    })
    weights_df.to_csv('models/ensemble_weights.csv', index=False)
    
    logger.info("Models saved to 'models/' directory")
    
    # Save the feature engineering pipeline for future use
    # Create a dictionary with feature transformations
    feature_engineering_steps = {
        'create_BMI': lambda df: df['Weight'] / ((df['Height']/100) ** 2),
        'create_Exercise_Intensity': lambda df: df['Duration'] * df['Heart_Rate'],
        'create_Thermal_Load': lambda df: df['Duration'] * df['Body_Temp'],
        'create_Work_Load': lambda df: df['Weight'] * df['Duration'],
        'create_Heart_Rate_Reserve': lambda df: df['Heart_Rate'] / (220 - df['Age']),
        'create_MET_Approx': lambda df: (df['Heart_Rate'] - 60) / 10 + 1,
        'create_Energy_Score': lambda df: df['Duration'] * df['Heart_Rate'] * df['Body_Temp'] / 100
    }
    
    # Save the feature engineering steps
    joblib.dump(feature_engineering_steps, 'models/feature_engineering_steps.pkl')
    
    # Sample code for applying the ensemble to new data
    with open('models/prediction_code_sample.py', 'w') as f:
        f.write('''
# Sample code for making predictions with the ensemble model
import joblib
import pandas as pd
import numpy as np

def predict_calories(new_data_path):
    """
    Apply the ensemble model to new data
    
    Parameters:
    -----------
    new_data_path : str
        Path to the CSV file with new data (must have same columns as training data)
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with predicted calories
    """
    # Load the models
    rf_model = joblib.load('models/random_forest_optimized.pkl')
    xgb_model = joblib.load('models/xgboost_optimized.pkl')
    lgb_model = joblib.load('models/lightgbm_optimized.pkl')
    
    # Load the weights
    weights_df = pd.read_csv('models/ensemble_weights.csv')
    weights = weights_df['Weight'].tolist()
    
    # Load the feature engineering steps
    feature_steps = joblib.load('models/feature_engineering_steps.pkl')
    
    # Load new data
    new_data = pd.read_csv(new_data_path)
    
    # Feature engineering
    df_engineered = new_data.copy()
    
    # Apply feature engineering
    df_engineered['BMI'] = feature_steps['create_BMI'](df_engineered)
    df_engineered['Exercise_Intensity'] = feature_steps['create_Exercise_Intensity'](df_engineered)
    df_engineered['Thermal_Load'] = feature_steps['create_Thermal_Load'](df_engineered)
    df_engineered['Work_Load'] = feature_steps['create_Work_Load'](df_engineered)
    df_engineered['Heart_Rate_Reserve'] = feature_steps['create_Heart_Rate_Reserve'](df_engineered)
    df_engineered['MET_Approx'] = feature_steps['create_MET_Approx'](df_engineered)
    df_engineered['Energy_Score'] = feature_steps['create_Energy_Score'](df_engineered)
    
    # Add Sex-based features
    df_engineered['Sex_Numeric'] = df_engineered['Sex'].map({'male': 1, 'female': 0})
    df_engineered['Sex_Weight'] = df_engineered['Sex_Numeric'] * df_engineered['Weight']
    df_engineered['Sex_Height'] = df_engineered['Sex_Numeric'] * df_engineered['Height']
    df_engineered['Sex_BMI'] = df_engineered['Sex_Numeric'] * df_engineered['BMI']
    
    # Prepare the features for prediction
    X = df_engineered.drop(['id'], axis=1)
    if 'Calories' in X.columns:
        X = X.drop(['Calories'], axis=1)
    
    # Get dummy variables for categorical features
    X = pd.get_dummies(X, columns=['Sex'], drop_first=False)
    
    # Handle missing columns that model might expect
    for model in [rf_model, xgb_model, lgb_model]:
        missing_cols = set(model.feature_names_in_) - set(X.columns)
        # Add missing columns (with zeros)
        for col in missing_cols:
            X[col] = 0
        # Reorder columns to match model's expectation
        X = X[model.feature_names_in_]
    
    # Make predictions with each model
    rf_pred = rf_model.predict(X)
    xgb_pred = xgb_model.predict(X)
    lgb_pred = lgb_model.predict(X)
    
    # Weighted ensemble prediction
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
    
    return predictions_df

# Example usage
if __name__ == "__main__":
    # Replace with path to your test data
    test_data_path = 'playground-series-s5e5/test.csv'
    predictions = predict_calories(test_data_path)
    
    # Save predictions
    predictions.to_csv('predictions.csv', index=False)
    
    # For Kaggle submission
    submission = predictions[['id', 'Calories_Ensemble']].rename(columns={'Calories_Ensemble': 'Calories'})
    submission.to_csv('submission.csv', index=False)
    
    print("Predictions saved to 'predictions.csv' and 'submission.csv'")
        ''')
    
    logger.info("Created sample prediction code in 'models/prediction_code_sample.py'")
    logger.info("\nFeature engineering, hyperparameter tuning and ensemble modeling completed successfully!")

except Exception as e:
    logger.error(f"Error occurred: {str(e)}", exc_info=True)
    print(f"An error occurred: {str(e)}")