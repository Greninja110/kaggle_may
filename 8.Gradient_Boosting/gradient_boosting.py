# In VS Code, save this as gradient_boosting_models.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgb
import logging
import time
import os

# Create logs directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

# Set up logging with timestamp in filename
timestamp = time.strftime("%Y%m%d-%H%M%S")
log_filename = f'logs/gradient_boosting_{timestamp}.log'

logging.basicConfig(filename=log_filename, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Add console handler to display logs in terminal
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

try:
    # Load the data
    logger.info("Loading data...")
    data = pd.read_csv('playground-series-s5e5/train.csv')
    
    # Display basic info
    logger.info(f"Dataset shape: {data.shape}")
    logger.info(f"Dataset columns: {data.columns.tolist()}")
    logger.info(f"Dataset info:\n{data.dtypes}")
    
    # Display a sample of the data for debugging
    logger.info(f"Sample data:\n{data.head()}")
    
    # Check for missing values
    missing_values = data.isnull().sum()
    logger.info(f"Missing values per column:\n{missing_values}")
    
    # Preparing features and target
    X = data.drop(['id', 'Calories'], axis=1)
    y = data['Calories']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info(f"Training set size: {X_train.shape}, Test set size: {X_test.shape}")
    
    # Preprocessing for categorical variable (Sex)
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first'), ['Sex'])
        ],
        remainder='passthrough'
    )
    
    # Create results dictionary to store model metrics
    results = {}
    
    # 1. XGBoost Model
    logger.info("Training XGBoost model...")
    xgb_model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', xgb.XGBRegressor(
            n_estimators=100, 
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            n_jobs=-1  # Use all available cores
        ))
    ])
    
    # Train the XGBoost model with timing
    start_time = time.time()
    xgb_model.fit(X_train, y_train)
    xgb_training_time = time.time() - start_time
    logger.info(f"XGBoost training completed in {xgb_training_time:.2f} seconds")
    
    # Make predictions with XGBoost
    xgb_y_pred = xgb_model.predict(X_test)
    
    # Evaluate the XGBoost model
    xgb_mse = mean_squared_error(y_test, xgb_y_pred)
    xgb_rmse = np.sqrt(xgb_mse)
    xgb_r2 = r2_score(y_test, xgb_y_pred)
    
    logger.info(f"XGBoost Model performance:")
    logger.info(f"Mean Squared Error: {xgb_mse:.2f}")
    logger.info(f"Root Mean Squared Error: {xgb_rmse:.2f}")
    logger.info(f"R² Score: {xgb_r2:.2f}")
    
    results['XGBoost'] = {
        'MSE': xgb_mse,
        'RMSE': xgb_rmse,
        'R2': xgb_r2,
        'Training Time': xgb_training_time,
        'Predictions': xgb_y_pred
    }
    
    # 2. LightGBM Model
    logger.info("Training LightGBM model...")
    lgb_model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            n_jobs=-1  # Use all available cores
        ))
    ])
    
    # Train the LightGBM model with timing
    start_time = time.time()
    lgb_model.fit(X_train, y_train)
    lgb_training_time = time.time() - start_time
    logger.info(f"LightGBM training completed in {lgb_training_time:.2f} seconds")
    
    # Make predictions with LightGBM
    lgb_y_pred = lgb_model.predict(X_test)
    
    # Evaluate the LightGBM model
    lgb_mse = mean_squared_error(y_test, lgb_y_pred)
    lgb_rmse = np.sqrt(lgb_mse)
    lgb_r2 = r2_score(y_test, lgb_y_pred)
    
    logger.info(f"LightGBM Model performance:")
    logger.info(f"Mean Squared Error: {lgb_mse:.2f}")
    logger.info(f"Root Mean Squared Error: {lgb_rmse:.2f}")
    logger.info(f"R² Score: {lgb_r2:.2f}")
    
    results['LightGBM'] = {
        'MSE': lgb_mse,
        'RMSE': lgb_rmse,
        'R2': lgb_r2,
        'Training Time': lgb_training_time,
        'Predictions': lgb_y_pred
    }
    
    # Compare models
    logger.info("Model Comparison:")
    for model_name, metrics in results.items():
        logger.info(f"{model_name} - RMSE: {metrics['RMSE']:.2f}, R²: {metrics['R2']:.2f}, Training Time: {metrics['Training Time']:.2f}s")
    
    # Determine best model based on RMSE
    best_model = min(results.items(), key=lambda x: x[1]['RMSE'])
    logger.info(f"Best model based on RMSE: {best_model[0]} with RMSE of {best_model[1]['RMSE']:.2f}")
    
    # Print results to console as well
    print("\nModel Comparison:")
    for model_name, metrics in results.items():
        print(f"{model_name} - RMSE: {metrics['RMSE']:.2f}, R²: {metrics['R2']:.2f}, Training Time: {metrics['Training Time']:.2f}s")
    print(f"Best model based on RMSE: {best_model[0]} with RMSE of {best_model[1]['RMSE']:.2f}")
    
    # Visualize actual vs predicted values for both models
    plt.figure(figsize=(15, 7))
    
    # XGBoost plot
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, results['XGBoost']['Predictions'], alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Calories')
    plt.ylabel('Predicted Calories')
    plt.title(f'XGBoost: Actual vs Predicted\nRMSE: {xgb_rmse:.2f}, R²: {xgb_r2:.2f}')
    
    # LightGBM plot
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, results['LightGBM']['Predictions'], alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Calories')
    plt.ylabel('Predicted Calories')
    plt.title(f'LightGBM: Actual vs Predicted\nRMSE: {lgb_rmse:.2f}, R²: {lgb_r2:.2f}')
    
    plt.tight_layout()
    plt.savefig('gradient_boosting_results.png')
    logger.info("Saved visualization to 'gradient_boosting_results.png'")
    plt.show()
    
    # Feature importance analysis (only for XGBoost as example)
    logger.info("Analyzing feature importance...")
    
    # Get feature names after preprocessing
    feature_names = []
    # Add the binary feature name for Sex (assuming 'Sex' gets transformed to 'Sex_Male' or similar)
    feature_names.append('Sex_encoded')
    # Add the remaining feature names
    for col in X.columns:
        if col != 'Sex':  # Skip 'Sex' as it was already added in encoded form
            feature_names.append(col)
    
    # Get the regressor from the pipeline
    xgb_regressor = xgb_model.named_steps['regressor']
    
    # Get feature importance
    try:
        importance = xgb_regressor.feature_importances_
        
        # Create a DataFrame for better visualization
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values(by='Importance', ascending=False)
        
        logger.info(f"Feature importance:\n{feature_importance}")
        
        # Visualize feature importance
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance['Feature'], feature_importance['Importance'])
        plt.xlabel('Importance')
        plt.title('XGBoost Feature Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        logger.info("Saved feature importance visualization to 'feature_importance.png'")
        plt.show()
    except Exception as e:
        logger.warning(f"Could not calculate feature importance: {str(e)}")
    
    # Save models for future use
    import joblib
    joblib.dump(xgb_model, 'xgboost_model.pkl')
    joblib.dump(lgb_model, 'lightgbm_model.pkl')
    logger.info("Saved models to disk as 'xgboost_model.pkl' and 'lightgbm_model.pkl'")
    
except Exception as e:
    logger.error(f"Error occurred: {str(e)}", exc_info=True)
    print(f"An error occurred: {str(e)}")