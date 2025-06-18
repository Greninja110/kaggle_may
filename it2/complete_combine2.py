# In VS Code or Kaggle notebook, save this as enhanced_calorie_prediction.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import logging
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
import xgboost as xgb
import lightgbm as lgb
import optuna  # For advanced hyperparameter tuning
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
    
    # 5. Heart Rate Reserve
    df_engineered['Heart_Rate_Reserve'] = create_Heart_Rate_Reserve(df_engineered)
    
    # 6. BMI squares and cubes (non-linear relationships)
    df_engineered['BMI_Squared'] = df_engineered['BMI'] ** 2
    df_engineered['BMI_Cubed'] = df_engineered['BMI'] ** 3
    
    # 7. Sex-based interactions
    df_engineered['Sex_Numeric'] = df_engineered['Sex'].map({'male': 1, 'female': 0})
    df_engineered['Sex_Weight'] = df_engineered['Sex_Numeric'] * df_engineered['Weight']
    df_engineered['Sex_Height'] = df_engineered['Sex_Numeric'] * df_engineered['Height']
    df_engineered['Sex_BMI'] = df_engineered['Sex_Numeric'] * df_engineered['BMI']
    
    # 8. MET approximation
    df_engineered['MET_Approx'] = create_MET_Approx(df_engineered)
    
    # 9. Energy Expenditure Score
    df_engineered['Energy_Score'] = create_Energy_Score(df_engineered)
    
    # 10. Heart rate zones (physiological feature)
    df_engineered['HR_Zone'] = pd.cut(
        df_engineered['Heart_Rate_Reserve'], 
        bins=[0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 
        labels=['Zone1', 'Zone2', 'Zone3', 'Zone4', 'Zone5', 'Zone6']
    )
    
    # 11. Age-based training impacts
    df_engineered['Age_Exercise_Impact'] = df_engineered['Age'] * df_engineered['Exercise_Intensity'] / 1000
    
    # 12. Weight to Height ratio
    df_engineered['Weight_Height_Ratio'] = df_engineered['Weight'] / df_engineered['Height']
    
    # 13. Temperature-adjusted intensity
    df_engineered['Temp_Adjusted_Intensity'] = df_engineered['Exercise_Intensity'] * (df_engineered['Body_Temp'] / 37.0)
    
    # Age categories (for easier handling)
    bins = [0, 30, 50, 100]
    labels = ['Young', 'Middle_Aged', 'Senior']
    df_engineered['Age_Group'] = pd.cut(df_engineered['Age'], bins=bins, labels=labels)
    
    # BMI categories (for easier handling)
    bmi_bins = [0, 18.5, 25, 30, 100]
    bmi_labels = ['Underweight', 'Normal', 'Overweight', 'Obese']
    df_engineered['BMI_Category'] = pd.cut(df_engineered['BMI'], bins=bmi_bins, labels=bmi_labels)
    
    return df_engineered

def find_best_data_path():
    """Find the correct path for the dataset files in Kaggle"""
    possible_paths = [
        '/kaggle/input/kaggle-s5e5',
        '/kaggle/input/playground-series-s5e5',
        'playground-series-s5e5',
        '.'
    ]
    
    for path in possible_paths:
        train_path = os.path.join(path, 'train.csv')
        test_path = os.path.join(path, 'test.csv')
        
        if os.path.exists(train_path) and os.path.exists(test_path):
            return path
    
    # If we can't find the data, try scanning all input paths
    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            if filename == 'train.csv':
                return os.path.dirname(os.path.join(dirname, filename))
    
    return None

def optimize_xgboost(trial, X_train, X_test, y_train, y_test):
    """Optuna objective function for XGBoost parameter optimization"""
    param = {
        'objective': 'reg:squarederror',
        'tree_method': 'gpu_hist',  # Use GPU acceleration
        'gpu_id': trial.suggest_categorical('gpu_id', [0, 1]),  # Alternate between GPUs
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
    }
    
    # Create validation set for early stopping
    X_train_sub, X_valid, y_train_sub, y_valid = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Create DMatrix objects for XGBoost
    dtrain = xgb.DMatrix(X_train_sub, label=y_train_sub)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)
    
    # Train XGBoost with early stopping
    model = xgb.train(
        param,
        dtrain,
        num_boost_round=param['n_estimators'],
        evals=[(dvalid, 'validation')],
        early_stopping_rounds=50,
        verbose_eval=False
    )
    
    # Predict on test data
    dtest = xgb.DMatrix(X_test)
    y_pred = model.predict(dtest)
    
    # Calculate RMSLE
    score = rmsle(y_test, y_pred)
    
    return score

def optimize_lightgbm(trial, X_train, X_test, y_train, y_test):
    """Optuna objective function for LightGBM parameter optimization"""
    param = {
        'objective': 'regression',
        'device': 'gpu',  # Use GPU
        'gpu_platform_id': 0,
        'gpu_device_id': trial.suggest_categorical('gpu_device_id', [0, 1]),  # Alternate between GPUs
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', -1, 20),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        'max_bin': trial.suggest_categorical('max_bin', [63, 127, 255]),
        'verbose': -1  # Control verbosity through param instead
    }
    
    # Create validation set for early stopping
    X_train_sub, X_valid, y_train_sub, y_valid = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train_sub, label=y_train_sub)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
    
    # Number of boosting rounds
    num_boost_round = trial.suggest_int('n_estimators', 100, 1000)
    
    # Train LightGBM with early stopping
    model = lgb.train(
        param,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
        # Remove verbose_eval parameter - it doesn't exist in LightGBM API
    )
    
    # Predict on test data
    y_pred = model.predict(X_test)
    
    # Calculate RMSLE
    score = rmsle(y_test, y_pred)
    
    return score

def optimize_random_forest(trial, X_train, X_test, y_train, y_test):
    """Optuna objective function for Random Forest parameter optimization"""
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'n_jobs': -1  # Use all CPUs
    }
    
    # Train Random Forest
    model = RandomForestRegressor(**param, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict on test data
    y_pred = model.predict(X_test)
    
    # Calculate RMSLE
    score = rmsle(y_test, y_pred)
    
    return score

def train_and_evaluate_models():
    """Advanced training with Optuna for hyperparameter optimization and GPU utilization"""
    try:
        logger.info("Starting advanced feature engineering and model optimization...")
        
        # Find the correct data path
        data_path = find_best_data_path()
        if data_path is None:
            raise FileNotFoundError("Could not find dataset files in any expected locations.")
        
        # Load the data
        logger.info(f"Loading data from {data_path}...")
        train_path = os.path.join(data_path, 'train.csv')
        data = pd.read_csv(train_path)
        logger.info(f"Successfully loaded data with shape: {data.shape}")
        
        # Check for missing values
        missing_values = data.isnull().sum()
        if missing_values.sum() > 0:
            logger.warning(f"Dataset contains {missing_values.sum()} missing values. Handling missing values...")
            data = data.fillna(data.median())
        
        # ===============================================================
        # FEATURE ENGINEERING 
        # ===============================================================
        logger.info("Performing advanced feature engineering...")
        
        # Apply feature engineering
        df_engineered = apply_feature_engineering(data)
        
        # Log the new features
        logger.info(f"New engineered features: {[col for col in df_engineered.columns if col not in data.columns]}")
        
        # Prepare features and target for modeling
        # Drop unnecessary columns and categorical columns that need to be encoded
        X = df_engineered.drop(['id', 'Calories', 'Age_Group', 'BMI_Category', 'HR_Zone'], axis=1)
        
        # Get dummy variables for categorical features
        X = pd.get_dummies(X, columns=['Sex'], drop_first=False)
        y = df_engineered['Calories']
        
        # Check for any remaining categorical columns
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_cols:
            logger.info(f"Encoding remaining categorical columns: {cat_cols}")
            X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
        
        # Scale numerical features for improved performance
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
        scaler = StandardScaler()
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
        
        logger.info(f"Final feature set shape: {X.shape}")
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logger.info(f"Training set size: {X_train.shape}, Test set size: {X_test.shape}")
        
        # ===============================================================
        # OPTUNA HYPERPARAMETER OPTIMIZATION
        # ===============================================================
        logger.info("Starting advanced hyperparameter optimization with Optuna...")
        
        # 1. XGBoost Hyperparameter Tuning
        logger.info("Optimizing XGBoost with GPU acceleration...")
        
        study_xgb = optuna.create_study(direction='minimize')
        study_xgb.optimize(lambda trial: optimize_xgboost(trial, X_train, X_test, y_train, y_test), 
                          n_trials=30, 
                          timeout=1800)  # 30 minutes timeout
        
        xgb_best_params = study_xgb.best_params
        xgb_best_value = study_xgb.best_value
        
        logger.info(f"XGBoost best parameters: {xgb_best_params}")
        logger.info(f"XGBoost best RMSLE score: {xgb_best_value:.6f}")
        
        # 2. LightGBM Hyperparameter Tuning
        logger.info("Optimizing LightGBM with GPU acceleration...")
        
        study_lgb = optuna.create_study(direction='minimize')
        study_lgb.optimize(lambda trial: optimize_lightgbm(trial, X_train, X_test, y_train, y_test), 
                          n_trials=30, 
                          timeout=1800)  # 30 minutes timeout
        
        lgb_best_params = study_lgb.best_params
        lgb_best_value = study_lgb.best_value
        
        logger.info(f"LightGBM best parameters: {lgb_best_params}")
        logger.info(f"LightGBM best RMSLE score: {lgb_best_value:.6f}")
        
        # 3. Random Forest Hyperparameter Tuning
        logger.info("Optimizing Random Forest...")
        
        study_rf = optuna.create_study(direction='minimize')
        study_rf.optimize(lambda trial: optimize_random_forest(trial, X_train, X_test, y_train, y_test), 
                         n_trials=30, 
                         timeout=1800)  # 30 minutes timeout
        
        rf_best_params = study_rf.best_params
        rf_best_value = study_rf.best_value
        
        logger.info(f"Random Forest best parameters: {rf_best_params}")
        logger.info(f"Random Forest best RMSLE score: {rf_best_value:.6f}")
        
        # ===============================================================
        # TRAIN FINAL MODELS WITH BEST PARAMETERS
        # ===============================================================
        logger.info("Training final models with optimized parameters...")
        
        # XGBoost final model with GPU acceleration
        best_xgb_params = {k: v for k, v in xgb_best_params.items() if k != 'n_estimators'}
        best_xgb_params.update({
            'objective': 'reg:squarederror',
            'tree_method': 'gpu_hist',
            'gpu_id': 0  # Use first GPU for final training
        })
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        best_xgb = xgb.train(
            best_xgb_params,
            dtrain,
            num_boost_round=xgb_best_params.get('n_estimators', 500),
            evals=[(dtest, 'test')],
            early_stopping_rounds=50,
            verbose_eval=100
        )
        
        xgb_pred = best_xgb.predict(dtest)
        xgb_rmsle = rmsle(y_test, xgb_pred)
        logger.info(f"Final XGBoost RMSLE on test set: {xgb_rmsle:.6f}")
        
        # LightGBM final model with GPU acceleration
        best_lgb_params = {k: v for k, v in lgb_best_params.items() if k != 'n_estimators'}
        best_lgb_params.update({
            'objective': 'regression',
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 1  # Use second GPU for final training
        })
        
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        best_lgb = lgb.train(
            best_lgb_params,
            train_data,
            num_boost_round=lgb_best_params.get('n_estimators', 500),
            valid_sets=[test_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
        )
        
        lgb_pred = best_lgb.predict(X_test)
        lgb_rmsle = rmsle(y_test, lgb_pred)
        logger.info(f"Final LightGBM RMSLE on test set: {lgb_rmsle:.6f}")
        
        # Random Forest final model
        best_rf = RandomForestRegressor(**rf_best_params, random_state=42)
        best_rf.fit(X_train, y_train)
        
        rf_pred = best_rf.predict(X_test)
        rf_rmsle = rmsle(y_test, rf_pred)
        logger.info(f"Final Random Forest RMSLE on test set: {rf_rmsle:.6f}")
        
        # ===============================================================
        # STACKING ENSEMBLE
        # ===============================================================
        logger.info("Creating stacking ensemble model...")
        
        # Define base estimators
        estimators = [
            ('rf', RandomForestRegressor(**rf_best_params, random_state=42)),
            ('xgb', xgb.XGBRegressor(
                objective='reg:squarederror',
                tree_method='gpu_hist',
                gpu_id=0,
                **{k: v for k, v in xgb_best_params.items() if k != 'gpu_id'}
            )),
            ('lgb', lgb.LGBMRegressor(
                objective='regression',
                device='gpu',
                gpu_platform_id=0,
                gpu_device_id=1,
                **{k: v for k, v in lgb_best_params.items() if k not in ['gpu_device_id', 'gpu_platform_id']}
            ))
        ]
        
        # Create stacking ensemble with Ridge as the final estimator
        stacking_model = StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(alpha=1.0),
            cv=5,
            n_jobs=1
        )
        
        # Train stacking ensemble
        stacking_model.fit(X_train, y_train)
        stack_pred = stacking_model.predict(X_test)
        stack_rmsle = rmsle(y_test, stack_pred)
        logger.info(f"Stacking Ensemble RMSLE on test set: {stack_rmsle:.6f}")
        
        # ===============================================================
        # WEIGHTED ENSEMBLE
        # ===============================================================
        logger.info("Creating weighted ensemble...")
        
        # Calculate weights based on inverse of RMSLE (lower RMSLE = higher weight)
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
        
        ensemble_rmsle = rmsle(y_test, ensemble_pred)
        ensemble_mse = mean_squared_error(y_test, ensemble_pred)
        ensemble_r2 = r2_score(y_test, ensemble_pred)
        
        logger.info("Weighted Ensemble model performance:")
        logger.info(f"RMSLE (competition metric): {ensemble_rmsle:.6f}")
        logger.info(f"MSE: {ensemble_mse:.6f}")
        logger.info(f"R² Score: {ensemble_r2:.6f}")
        
        # ===============================================================
        # COMPARE ALL MODELS
        # ===============================================================
        # Compare individual models with both ensembles
        results_df = pd.DataFrame({
            'Model': ['Random Forest', 'XGBoost', 'LightGBM', 'Stacking Ensemble', 'Weighted Ensemble'],
            'RMSLE': [rf_rmsle, xgb_rmsle, lgb_rmsle, stack_rmsle, ensemble_rmsle],
            'MSE': [
                mean_squared_error(y_test, rf_pred), 
                mean_squared_error(y_test, xgb_pred), 
                mean_squared_error(y_test, lgb_pred),
                mean_squared_error(y_test, stack_pred),
                ensemble_mse
            ],
            'R²': [
                r2_score(y_test, rf_pred), 
                r2_score(y_test, xgb_pred), 
                r2_score(y_test, lgb_pred),
                r2_score(y_test, stack_pred),
                ensemble_r2
            ],
            'Weight': [weights[0], weights[1], weights[2], 'N/A', 'N/A']
        })
        
        logger.info("\nFinal Model Comparison:")
        logger.info(f"\n{results_df.to_string(index=False)}")
        
        # Determine the best model
        best_model_name = results_df.loc[results_df['RMSLE'].idxmin(), 'Model']
        best_model_rmsle = results_df['RMSLE'].min()
        logger.info(f"\nBest model: {best_model_name} with RMSLE = {best_model_rmsle:.6f}")
        
        # Save the results to a CSV file
        output_path = '/kaggle/working/model_comparison_results.csv' if os.path.exists('/kaggle') else 'model_comparison_results.csv'
        results_df.to_csv(output_path, index=False)
        
        # Create visualizations
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Model', y='RMSLE', data=results_df, palette='viridis')
        plt.xlabel('Models')
        plt.ylabel('RMSLE (Lower is Better)')
        plt.title('Model Comparison by RMSLE')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add RMSLE values on top of each bar
        for i, v in enumerate(results_df['RMSLE']):
            plt.text(i, v + 0.001, f'{v:.6f}', ha='center')
        
        plt.tight_layout()
        plot_path = '/kaggle/working/model_comparison.png' if os.path.exists('/kaggle') else 'plots/model_comparison.png'
        plt.savefig(plot_path)
        
        # Save models
        models_dir = '/kaggle/working/models' if os.path.exists('/kaggle') else 'models'
        os.makedirs(models_dir, exist_ok=True)
        
        # Save individual models
        joblib.dump(best_rf, f'{models_dir}/random_forest_optimized.pkl')
        best_xgb.save_model(f'{models_dir}/xgboost_optimized.model')
        best_lgb.save_model(f'{models_dir}/lightgbm_optimized.txt')
        joblib.dump(stacking_model, f'{models_dir}/stacking_ensemble.pkl')
        
        # Save the scaler for preprocessing
        joblib.dump(scaler, f'{models_dir}/scaler.pkl')
        
        # Store weights
        weights_df = pd.DataFrame({
            'Model': ['RandomForest', 'XGBoost', 'LightGBM'],
            'Weight': weights
        })
        weights_df.to_csv(f'{models_dir}/ensemble_weights.csv', index=False)
        
        logger.info(f"Models saved to '{models_dir}' directory")
        
        # Return the best model information for prediction
        if best_model_name == 'Stacking Ensemble':
            return stacking_model, None, None, None, X.columns, scaler
        elif best_model_name == 'Weighted Ensemble':
            return best_rf, best_xgb, best_lgb, weights, X.columns, scaler
        elif best_model_name == 'Random Forest':
            return best_rf, None, None, None, X.columns, scaler
        elif best_model_name == 'XGBoost':
            return None, best_xgb, None, None, X.columns, scaler
        else:  # LightGBM
            return None, None, best_lgb, None, X.columns, scaler
        
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}", exc_info=True)
        print(f"An error occurred: {str(e)}")
        return None, None, None, None, None, None

def predict_and_create_submission():
    """
    Generate predictions for the test data and create a submission file
    """
    try:
        # Find the correct data path
        data_path = find_best_data_path()
        if data_path is None:
            raise FileNotFoundError("Could not find dataset files in any expected locations.")
        
        # Check for saved models or train new ones
        models_dir = '/kaggle/working/models' if os.path.exists('/kaggle') else 'models'
        
        if (os.path.exists(f'{models_dir}/random_forest_optimized.pkl') and
            os.path.exists(f'{models_dir}/xgboost_optimized.model') and
            os.path.exists(f'{models_dir}/lightgbm_optimized.txt') and
            os.path.exists(f'{models_dir}/ensemble_weights.csv')):
            
            logger.info("Loading saved models...")
            rf_model = joblib.load(f'{models_dir}/random_forest_optimized.pkl')
            xgb_model = xgb.Booster()
            xgb_model.load_model(f'{models_dir}/xgboost_optimized.model')
            lgb_model = lgb.Booster(model_file=f'{models_dir}/lightgbm_optimized.txt')
            
            # Load weights
            weights_df = pd.read_csv(f'{models_dir}/ensemble_weights.csv')
            weights = weights_df['Weight'].tolist()
            
            # Load the scaler
            scaler = joblib.load(f'{models_dir}/scaler.pkl')
            
            # Check if stacking model exists and load it
            stacking_model = None
            if os.path.exists(f'{models_dir}/stacking_ensemble.pkl'):
                stacking_model = joblib.load(f'{models_dir}/stacking_ensemble.pkl')
                logger.info("Stacking ensemble model loaded")
            
            logger.info("Models loaded successfully")
            
        else:
            logger.info("No saved models found. Training new models...")
            rf_model, xgb_model, lgb_model, weights, feature_names, scaler = train_and_evaluate_models()
            stacking_model = None
            
            if rf_model is None and xgb_model is None and lgb_model is None:
                raise Exception("Model training failed")
        
        # Load test data
        logger.info("Loading test data...")
        test_path = os.path.join(data_path, 'test.csv')
        test_data = pd.read_csv(test_path)
        logger.info(f"Test data loaded with shape: {test_data.shape}")
        
        # Apply feature engineering to test data
        logger.info("Applying feature engineering to test data...")
        test_engineered = apply_feature_engineering(test_data)
        
        # Prepare test features for prediction
        X_test = test_engineered.drop(['id', 'Age_Group', 'BMI_Category', 'HR_Zone'], axis=1)
        
        # Get dummy variables for categorical features
        X_test = pd.get_dummies(X_test, columns=['Sex'], drop_first=False)
        
        # Handle any remaining categorical columns
        cat_cols = X_test.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_cols:
            logger.info(f"Encoding remaining categorical columns: {cat_cols}")
            X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)
        
        # Apply the scaler to numerical features
        numerical_cols = X_test.select_dtypes(include=['int64', 'float64']).columns
        X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
        
        logger.info(f"Prepared test features with shape: {X_test.shape}")
        
        # Make predictions
        logger.info("Making predictions...")
        
        # Check if we have a stacking model
        if stacking_model is not None:
            logger.info("Using stacking ensemble for predictions...")
            ensemble_pred = stacking_model.predict(X_test)
        else:
            # Make predictions with individual models
            predictions = []
            
            # Random Forest prediction (if available)
            if rf_model is not None:
                logger.info("Generating Random Forest predictions...")
                rf_pred = rf_model.predict(X_test)
                predictions.append((rf_pred, weights[0] if weights else 1))
            
            # XGBoost prediction (if available)
            if xgb_model is not None:
                logger.info("Generating XGBoost predictions...")
                dtest = xgb.DMatrix(X_test)
                xgb_pred = xgb_model.predict(dtest)
                predictions.append((xgb_pred, weights[1] if weights else 1))
            
            # LightGBM prediction (if available)
            if lgb_model is not None:
                logger.info("Generating LightGBM predictions...")
                lgb_pred = lgb_model.predict(X_test)
                predictions.append((lgb_pred, weights[2] if weights else 1))
            
            # Combine predictions based on weights
            if len(predictions) > 1 and weights:
                logger.info(f"Creating weighted ensemble prediction with {len(predictions)} models...")
                ensemble_pred = sum(pred * weight for pred, weight in predictions)
            else:
                # Just use the available model's prediction
                ensemble_pred = predictions[0][0]
        
        # Create Kaggle submission file
        submission = pd.DataFrame({
            'id': test_data['id'],
            'Calories': ensemble_pred
        })
        
        submission_path = '/kaggle/working/submission.csv' if os.path.exists('/kaggle') else 'submission.csv'
        submission.to_csv(submission_path, index=False)
        logger.info(f"Submission file saved to '{submission_path}'")
        
        # Create detailed prediction file with statistics
        if len(predictions) > 1:
            detailed_pred_df = pd.DataFrame({'id': test_data['id']})
            
            model_names = []
            for i, (pred, _) in enumerate(predictions):
                model_name = ['Random Forest', 'XGBoost', 'LightGBM'][i]
                detailed_pred_df[f'Calories_{model_name}'] = pred
                model_names.append(model_name)
            
            detailed_pred_df['Calories_Ensemble'] = ensemble_pred
            
            # Calculate basic statistics
            stats_df = pd.DataFrame({
                'Model': model_names + ['Ensemble'],
                'Mean': [detailed_pred_df[f'Calories_{name}'].mean() for name in model_names] + [ensemble_pred.mean()],
                'Min': [detailed_pred_df[f'Calories_{name}'].min() for name in model_names] + [ensemble_pred.min()],
                'Max': [detailed_pred_df[f'Calories_{name}'].max() for name in model_names] + [ensemble_pred.max()],
                'Std': [detailed_pred_df[f'Calories_{name}'].std() for name in model_names] + [ensemble_pred.std()]
            })
            
            # Save detailed predictions and stats
            detailed_output = '/kaggle/working/detailed_predictions.csv' if os.path.exists('/kaggle') else 'detailed_predictions.csv'
            detailed_pred_df.to_csv(detailed_output, index=False)
            
            stats_output = '/kaggle/working/prediction_stats.csv' if os.path.exists('/kaggle') else 'prediction_stats.csv'
            stats_df.to_csv(stats_output, index=False)
            
            logger.info(f"Detailed predictions saved to '{detailed_output}'")
            logger.info(f"Prediction statistics saved to '{stats_output}'")
            
            # Print summary statistics
            logger.info("\nPrediction Summary Statistics:")
            logger.info(f"\n{stats_df.to_string(index=False)}")
        
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