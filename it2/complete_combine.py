# In VS Code or Kaggle notebook, save this as calorie_prediction_ensemble.py
# didnt work , 19k seconds in kaggle , so edited , to complete_combine2.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
import xgboost as xgb
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')

# Create directories
os.makedirs('logs', exist_ok=True)
os.makedirs('plots', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Set up logging
log_filename = f'logs/calorie_prediction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Feature engineering functions
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

# RMSLE (Root Mean Squared Logarithmic Error) - Competition Metric
def rmsle(y_true, y_pred):
    """
    Calculate Root Mean Squared Logarithmic Error
    RMSLE = sqrt(1/n * sum((log(1+y_true) - log(1+y_pred))^2))
    """
    # Ensure predictions are positive (since we're taking logs)
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(np.mean(np.power(np.log1p(y_true) - np.log1p(y_pred), 2)))

# Create a scorer for cross-validation
rmsle_scorer = make_scorer(rmsle, greater_is_better=False)

def apply_feature_engineering(df):
    """Apply all feature engineering transformations to the DataFrame"""
    df_engineered = df.copy()
    
    # 1. BMI (Body Mass Index)
    df_engineered['BMI'] = create_BMI(df_engineered)
    
    # 2. Exercise Intensity
    df_engineered['Exercise_Intensity'] = create_Exercise_Intensity(df_engineered)
    
    # 3. Thermal_Load
    df_engineered['Thermal_Load'] = create_Thermal_Load(df_engineered)
    
    # 4. Work_Load
    df_engineered['Work_Load'] = create_Work_Load(df_engineered)
    
    # 5. Age categories
    bins = [0, 30, 50, 100]
    labels = ['Young', 'Middle_Aged', 'Senior']
    df_engineered['Age_Group'] = pd.cut(df_engineered['Age'], bins=bins, labels=labels)
    
    # 6. BMI categories
    bmi_bins = [0, 18.5, 25, 30, 100]
    bmi_labels = ['Underweight', 'Normal', 'Overweight', 'Obese']
    df_engineered['BMI_Category'] = pd.cut(df_engineered['BMI'], bins=bmi_bins, labels=bmi_labels)
    
    # 7. Heart Rate Reserve
    df_engineered['Heart_Rate_Reserve'] = create_Heart_Rate_Reserve(df_engineered)
    
    # 8. Sex-based interactions
    df_engineered['Sex_Numeric'] = df_engineered['Sex'].map({'male': 1, 'female': 0})
    df_engineered['Sex_Weight'] = df_engineered['Sex_Numeric'] * df_engineered['Weight']
    df_engineered['Sex_Height'] = df_engineered['Sex_Numeric'] * df_engineered['Height']
    df_engineered['Sex_BMI'] = df_engineered['Sex_Numeric'] * df_engineered['BMI']
    
    # 9. MET approximation
    df_engineered['MET_Approx'] = create_MET_Approx(df_engineered)
    
    # 10. Energy Expenditure Score
    df_engineered['Energy_Score'] = create_Energy_Score(df_engineered)
    
    return df_engineered

def train_and_evaluate_models():
    """Train, tune, and evaluate the models, then create an ensemble"""
    try:
        logger.info("Starting feature engineering and ensemble modeling...")
        
        # Load the data
        logger.info("Loading data...")
        # Kaggle paths
        try:
            # In Kaggle notebook
            data = pd.read_csv('/kaggle/input/kaggle-s5e5/train.csv')
            logger.info("Successfully loaded data from Kaggle path")
        except FileNotFoundError:
            try:
                # Local paths
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
        
        # Check for missing values
        missing_values = data.isnull().sum()
        if missing_values.sum() > 0:
            logger.warning(f"Dataset contains {missing_values.sum()} missing values. Handling missing values...")
            data = data.fillna(data.median())
        
        # ===============================================================
        # FEATURE ENGINEERING 
        # ===============================================================
        logger.info("Performing feature engineering...")
        
        # Apply feature engineering
        df_engineered = apply_feature_engineering(data)
        
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
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        
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
        best_rf = RandomForestRegressor(**rf_best_params, random_state=42, n_jobs=-1)
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
        
        # Check if GPU is available for XGBoost
        try:
            # This will use GPU if available, fall back to CPU if not
            xgb_reg = xgb.XGBRegressor(
                objective='reg:squarederror',
                tree_method='gpu_hist',
                gpu_id=0,
                random_state=42
            )
            logger.info("XGBoost is using GPU acceleration")
        except Exception:
            xgb_reg = xgb.XGBRegressor(
                objective='reg:squarederror',
                random_state=42
            )
            logger.info("XGBoost is using CPU")
        
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
        
        # Create the best XGBoost model with GPU if available
        try:
            best_xgb = xgb.XGBRegressor(
                **xgb_best_params, 
                objective='reg:squarederror', 
                tree_method='gpu_hist',
                gpu_id=0,
                random_state=42
            )
        except Exception:
            best_xgb = xgb.XGBRegressor(
                **xgb_best_params, 
                objective='reg:squarederror', 
                random_state=42
            )
            
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
        
        # Check if GPU is available for LightGBM
        try:
            # This will use GPU if available
            lgb_reg = lgb.LGBMRegressor(
                objective='regression', 
                device='gpu',
                random_state=42
            )
            logger.info("LightGBM is using GPU acceleration")
        except Exception:
            lgb_reg = lgb.LGBMRegressor(
                objective='regression',
                random_state=42
            )
            logger.info("LightGBM is using CPU")
        
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
        
        # Create the best LightGBM model with GPU if available
        try:
            best_lgb = lgb.LGBMRegressor(
                **lgb_best_params, 
                objective='regression', 
                device='gpu',
                random_state=42
            )
        except Exception:
            best_lgb = lgb.LGBMRegressor(
                **lgb_best_params, 
                objective='regression',
                random_state=42
            )
            
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
        output_path = '/kaggle/working/model_comparison_results.csv' if os.path.exists('/kaggle') else 'model_comparison_results.csv'
        results_df.to_csv(output_path, index=False)
        
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
        plot_path = '/kaggle/working/ensemble_comparison.png' if os.path.exists('/kaggle') else 'plots/ensemble_comparison.png'
        plt.savefig(plot_path)
        
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
        features_plot_path = '/kaggle/working/feature_importance.png' if os.path.exists('/kaggle') else 'plots/feature_importance.png'
        plt.savefig(features_plot_path)
        
        # Save the best models for later use
        models_dir = '/kaggle/working/models' if os.path.exists('/kaggle') else 'models'
        os.makedirs(models_dir, exist_ok=True)
        
        joblib.dump(best_rf, f'{models_dir}/random_forest_optimized.pkl')
        joblib.dump(best_xgb, f'{models_dir}/xgboost_optimized.pkl')
        joblib.dump(best_lgb, f'{models_dir}/lightgbm_optimized.pkl')
        
        # Store weights
        weights_df = pd.DataFrame({
            'Model': ['RandomForest', 'XGBoost', 'LightGBM'],
            'Weight': weights
        })
        weights_df.to_csv(f'{models_dir}/ensemble_weights.csv', index=False)
        
        logger.info(f"Models saved to '{models_dir}' directory")
        
        # Store feature engineering functions
        feature_engineering_steps = {
            'create_BMI': create_BMI,
            'create_Exercise_Intensity': create_Exercise_Intensity,
            'create_Thermal_Load': create_Thermal_Load,
            'create_Work_Load': create_Work_Load,
            'create_Heart_Rate_Reserve': create_Heart_Rate_Reserve,
            'create_MET_Approx': create_MET_Approx,
            'create_Energy_Score': create_Energy_Score
        }
        
        joblib.dump(feature_engineering_steps, f'{models_dir}/feature_engineering_steps.pkl')
        
        # Return the models and weights for prediction
        return best_rf, best_xgb, best_lgb, weights, X.columns
        
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}", exc_info=True)
        print(f"An error occurred: {str(e)}")
        return None, None, None, None, None

def predict_and_create_submission():
    """
    Generate predictions for the test data and create a submission file
    """
    try:
        # Check for saved models or train new ones
        models_dir = '/kaggle/working/models' if os.path.exists('/kaggle') else 'models'
        
        if (os.path.exists(f'{models_dir}/random_forest_optimized.pkl') and
            os.path.exists(f'{models_dir}/xgboost_optimized.pkl') and
            os.path.exists(f'{models_dir}/lightgbm_optimized.pkl') and
            os.path.exists(f'{models_dir}/ensemble_weights.csv')):
            
            logger.info("Loading saved models...")
            best_rf = joblib.load(f'{models_dir}/random_forest_optimized.pkl')
            best_xgb = joblib.load(f'{models_dir}/xgboost_optimized.pkl')
            best_lgb = joblib.load(f'{models_dir}/lightgbm_optimized.pkl')
            
            # Load weights
            weights_df = pd.read_csv(f'{models_dir}/ensemble_weights.csv')
            weights = weights_df['Weight'].tolist()
            
            logger.info("Models loaded successfully")
        else:
            logger.info("No saved models found. Training new models...")
            best_rf, best_xgb, best_lgb, weights, _ = train_and_evaluate_models()
            if best_rf is None:
                raise Exception("Model training failed")
        
        # Load test data
        logger.info("Loading test data...")
        try:
            # For Kaggle
            test_data_path = '/kaggle/input/kaggle-s5e5/test.csv'
            test_data = pd.read_csv(test_data_path)
            logger.info(f"Successfully loaded test data from Kaggle path with shape: {test_data.shape}")
        except FileNotFoundError:
            try:
                # For local testing
                test_data_path = 'playground-series-s5e5/test.csv'
                test_data = pd.read_csv(test_data_path)
                logger.info(f"Successfully loaded test data from local path with shape: {test_data.shape}")
            except FileNotFoundError:
                try:
                    test_data_path = 'test.csv'
                    test_data = pd.read_csv(test_data_path)
                    logger.info(f"Successfully loaded test data from root path with shape: {test_data.shape}")
                except FileNotFoundError:
                    logger.error("Could not find the test data file. Please ensure it's in the correct location.")
                    raise
        
        # Apply feature engineering to test data
        logger.info("Applying feature engineering to test data...")
        test_engineered = apply_feature_engineering(test_data)
        
        # Prepare test features for prediction (drop unnecessary columns and encode categorical variables)
        X_test = test_engineered.drop(['id', 'Age_Group', 'BMI_Category'], axis=1)
        if 'Calories' in X_test.columns:
            X_test = X_test.drop(['Calories'], axis=1)
        
        # Get dummy variables for categorical features
        X_test = pd.get_dummies(X_test, columns=['Sex'], drop_first=False)
        
        # Handle any remaining categorical columns
        cat_cols = X_test.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_cols:
            logger.info(f"Encoding remaining categorical columns: {cat_cols}")
            X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)
        
        logger.info(f"Prepared test features with shape: {X_test.shape}")
        
        # Make predictions
        logger.info("Making predictions with each model...")
        
        # Helper function to align features
        def align_features(X, model):
            """Align features to match the model's expected feature names"""
            model_features = model.feature_names_in_
            
            # Add missing columns
            for col in model_features:
                if col not in X.columns:
                    X[col] = 0
                    
            # Remove extra columns
            X = X[model_features]
            return X
        
        # Random Forest prediction
        X_rf = align_features(X_test.copy(), best_rf)
        rf_pred = best_rf.predict(X_rf)
        
        # XGBoost prediction
        X_xgb = align_features(X_test.copy(), best_xgb)
        xgb_pred = best_xgb.predict(X_xgb)
        
        # LightGBM prediction
        X_lgb = align_features(X_test.copy(), best_lgb)
        lgb_pred = best_lgb.predict(X_lgb)
        
        # Weighted ensemble prediction
        logger.info(f"Creating ensemble prediction with weights: RF={weights[0]:.4f}, XGB={weights[1]:.4f}, LGB={weights[2]:.4f}")
        ensemble_pred = (
            weights[0] * rf_pred + 
            weights[1] * xgb_pred + 
            weights[2] * lgb_pred
        )
        
        # Create detailed predictions dataframe
        predictions_df = pd.DataFrame({
            'id': test_data['id'],
            'Calories_RF': rf_pred,
            'Calories_XGB': xgb_pred,
            'Calories_LGB': lgb_pred,
            'Calories_Ensemble': ensemble_pred
        })
        
        # Save detailed predictions
        detailed_output = '/kaggle/working/detailed_predictions.csv' if os.path.exists('/kaggle') else 'detailed_predictions.csv'
        predictions_df.to_csv(detailed_output, index=False)
        logger.info(f"Detailed predictions saved to '{detailed_output}'")
        
        # Create Kaggle submission file (just id and Calories)
        submission = predictions_df[['id', 'Calories_Ensemble']].rename(columns={'Calories_Ensemble': 'Calories'})
        submission_path = '/kaggle/working/submission.csv' if os.path.exists('/kaggle') else 'submission.csv'
        submission.to_csv(submission_path, index=False)
        logger.info(f"Submission file saved to '{submission_path}'")
        
        # Print summary statistics
        logger.info("\nPrediction Summary Statistics:")
        logger.info(f"Random Forest mean: {predictions_df['Calories_RF'].mean():.2f}")
        logger.info(f"XGBoost mean: {predictions_df['Calories_XGB'].mean():.2f}")
        logger.info(f"LightGBM mean: {predictions_df['Calories_LGB'].mean():.2f}")
        logger.info(f"Ensemble mean: {predictions_df['Calories_Ensemble'].mean():.2f}")
        
        return submission
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}", exc_info=True)
        print(f"Error in prediction: {str(e)}")
        return None

if __name__ == "__main__":
    # Set start time for total execution time calculation
    start_time = time.time()
    
    # Run the entire pipeline
    submission = predict_and_create_submission()
    
    # Calculate total execution time
    execution_time = time.time() - start_time
    
    if submission is not None:
        logger.info(f"Pipeline completed successfully in {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
        print(f"Pipeline completed successfully! Submission file is ready.")
    else:
        logger.error("Pipeline failed. Check logs for details.")
        print("Pipeline failed. Check logs for details.")