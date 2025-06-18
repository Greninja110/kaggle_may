# In VS Code, save this as compare_regression_models.py
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import logging
from datetime import datetime

# Create a logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# Set up logging
log_filename = f'logs/regression_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
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

# Create a scorer for cross-validation
rmsle_scorer = make_scorer(rmsle, greater_is_better=False)

try:
    logger.info("Starting regression model comparison for Kaggle Playground Series S5E5...")
    logger.info("Evaluation metric: Root Mean Squared Logarithmic Error (RMSLE)")
    
    # Load the data
    logger.info("Loading data...")
    # Try different potential paths for the data
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
    logger.info(f"Dataset info:\n{data.dtypes}")
    
    # Display statistical summary
    logger.info(f"Dataset summary statistics:\n{data.describe()}")
    
    # Check for missing values
    missing_values = data.isnull().sum()
    logger.info(f"Missing values:\n{missing_values}")
    
    if missing_values.sum() > 0:
        logger.warning(f"Dataset contains {missing_values.sum()} missing values. Handling missing values...")
        # Simple imputation for demonstration purposes
        data = data.fillna(data.median())
        logger.info("Missing values filled with median values.")
    
    # Basic EDA
    logger.info("Performing basic exploratory data analysis...")
    
    # Target variable distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data['Calories'], kde=True)
    plt.title('Distribution of Calories (Target Variable)')
    plt.savefig('plots/calories_distribution.png')
    plt.close()
    logger.info("Saved target variable distribution plot to 'plots/calories_distribution.png'")
    
    # Correlation matrix
    plt.figure(figsize=(12, 10))
    correlation_matrix = data.select_dtypes(include=['int64', 'float64']).corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('plots/correlation_matrix.png')
    plt.close()
    logger.info("Saved correlation matrix to 'plots/correlation_matrix.png'")
    
    # Preparing features and target
    logger.info("Preparing features and target variables...")
    X = data.drop(['id', 'Calories'], axis=1)
    y = data['Calories']
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    logger.info(f"Categorical columns: {categorical_cols}")
    logger.info(f"Numerical columns: {numerical_cols}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info(f"Training set size: {X_train.shape}, Test set size: {X_test.shape}")
    
    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols),
            ('num', StandardScaler(), numerical_cols)
        ]
    )
    
    # Dictionary to store model results
    results = []
    
    # Function to evaluate model using cross-validation and test set
    def evaluate_model(name, model, include_cv=True, n_cv_folds=5):
        start_time = time.time()
        
        # Create pipeline
        model_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Fit the model
        logger.info(f"Training {name} model...")
        model_pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model_pipeline.predict(X_test)
        
        # Ensure predictions are non-negative for RMSLE calculation
        y_pred = np.maximum(y_pred, 0)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        rmsle_score = rmsle(y_test, y_pred)
        
        # Cross-validation if requested
        cv_score = np.nan
        if include_cv:
            logger.info(f"Performing {n_cv_folds}-fold cross-validation for {name}...")
            try:
                cv_scores = cross_val_score(
                    model_pipeline, 
                    X, y, 
                    cv=n_cv_folds, 
                    scoring=rmsle_scorer
                )
                cv_score = -np.mean(cv_scores)  # Negate because the scorer is negative
                logger.info(f"{n_cv_folds}-fold CV RMSLE: {cv_score:.4f}")
            except Exception as e:
                logger.warning(f"Cross-validation failed for {name}: {str(e)}")
        
        training_time = time.time() - start_time
        
        logger.info(f"{name} model performance:")
        logger.info(f"Mean Squared Error: {mse:.4f}")
        logger.info(f"Root Mean Squared Error: {rmse:.4f}")
        logger.info(f"R² Score: {r2:.4f}")
        logger.info(f"RMSLE (competition metric): {rmsle_score:.4f}")
        logger.info(f"Training time: {training_time:.2f} seconds")
        
        # Visualize actual vs predicted
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Actual Calories')
        plt.ylabel('Predicted Calories')
        plt.title(f'{name}: Actual vs Predicted Calories')
        plt.savefig(f'plots/{name.lower().replace(" ", "_")}_results.png')
        plt.close()
        
        # Store results
        results.append({
            'Model': name,
            'MSE': mse,
            'RMSE': rmse,
            'R²': r2,
            'RMSLE': rmsle_score,
            'CV RMSLE': cv_score,
            'Training Time (s)': training_time
        })
        
        return model_pipeline
    
    # Dictionary to store trained models
    trained_models = {}
    
    logger.info("=" * 50)
    logger.info("Starting model evaluation")
    logger.info("=" * 50)
    
    # 1. Simple Linear Regression
    logger.info("\n--- Simple Linear Regression ---")
    trained_models['Simple Linear Regression'] = evaluate_model(
        'Simple Linear Regression', 
        LinearRegression()
    )
    
    # 2. Ridge Regression (L2 regularization)
    logger.info("\n--- Ridge Regression ---")
    trained_models['Ridge Regression'] = evaluate_model(
        'Ridge Regression', 
        Ridge(alpha=1.0)
    )
    
    # 3. Lasso Regression (L1 regularization)
    logger.info("\n--- Lasso Regression ---")
    trained_models['Lasso Regression'] = evaluate_model(
        'Lasso Regression', 
        Lasso(alpha=0.1)
    )
    
    # 4. ElasticNet (combination of L1 and L2)
    logger.info("\n--- ElasticNet Regression ---")
    trained_models['ElasticNet'] = evaluate_model(
        'ElasticNet', 
        ElasticNet(alpha=0.1, l1_ratio=0.5)
    )
    
    # 5. Polynomial Regression
    logger.info("\n--- Polynomial Regression ---")
    poly_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('poly', PolynomialFeatures(degree=2)),
        ('model', LinearRegression())
    ])
    
    # Fit polynomial regression manually since it has a different structure
    start_time = time.time()
    poly_pipeline.fit(X_train, y_train)
    y_pred_poly = poly_pipeline.predict(X_test)
    y_pred_poly = np.maximum(y_pred_poly, 0)  # Ensure non-negative predictions
    
    mse_poly = mean_squared_error(y_test, y_pred_poly)
    rmse_poly = np.sqrt(mse_poly)
    r2_poly = r2_score(y_test, y_pred_poly)
    rmsle_score_poly = rmsle(y_test, y_pred_poly)
    training_time_poly = time.time() - start_time
    
    logger.info(f"Polynomial Regression model performance:")
    logger.info(f"Mean Squared Error: {mse_poly:.4f}")
    logger.info(f"Root Mean Squared Error: {rmse_poly:.4f}")
    logger.info(f"R² Score: {r2_poly:.4f}")
    logger.info(f"RMSLE (competition metric): {rmsle_score_poly:.4f}")
    logger.info(f"Training time: {training_time_poly:.2f} seconds")
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_poly, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Calories')
    plt.ylabel('Predicted Calories')
    plt.title('Polynomial Regression: Actual vs Predicted Calories')
    plt.savefig('plots/polynomial_regression_results.png')
    plt.close()
    
    results.append({
        'Model': 'Polynomial Regression',
        'MSE': mse_poly,
        'RMSE': rmse_poly,
        'R²': r2_poly,
        'RMSLE': rmsle_score_poly,
        'CV RMSLE': np.nan,  # Not calculating CV score for polynomial regression
        'Training Time (s)': training_time_poly
    })
    
    trained_models['Polynomial Regression'] = poly_pipeline
    
    # 6. Decision Tree Regressor
    logger.info("\n--- Decision Tree Regressor ---")
    trained_models['Decision Tree'] = evaluate_model(
        'Decision Tree', 
        DecisionTreeRegressor(random_state=42)
    )
    
    # 7. Random Forest Regressor
    logger.info("\n--- Random Forest Regressor ---")
    trained_models['Random Forest'] = evaluate_model(
        'Random Forest', 
        RandomForestRegressor(n_estimators=100, random_state=42)
    )
    
    # 8. XGBoost and LightGBM (with error handling if not installed)
    # XGBoost
    logger.info("\n--- XGBoost Regressor ---")
    try:
        import xgboost as xgb
        trained_models['XGBoost'] = evaluate_model(
            'XGBoost', 
            xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                random_state=42
            )
        )
    except ImportError:
        logger.warning("XGBoost not installed. Install with: pip install xgboost")
        logger.warning("Skipping XGBoost evaluation.")
    except Exception as e:
        logger.warning(f"Error with XGBoost: {str(e)}")
        logger.warning("Skipping XGBoost evaluation.")
    
    # LightGBM
    logger.info("\n--- LightGBM Regressor ---")
    try:
        import lightgbm as lgb
        trained_models['LightGBM'] = evaluate_model(
            'LightGBM', 
            lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                random_state=42
            )
        )
    except ImportError:
        logger.warning("LightGBM not installed. Install with: pip install lightgbm")
        logger.warning("Skipping LightGBM evaluation.")
    except Exception as e:
        logger.warning(f"Error with LightGBM: {str(e)}")
        logger.warning("Skipping LightGBM evaluation.")
    
    logger.info("=" * 50)
    logger.info("Model evaluation completed")
    logger.info("=" * 50)
    
    # Create results dataframe and sort by RMSLE (ascending)
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('RMSLE')
    
    # Print results table
    logger.info("\nModel Comparison Results (sorted by RMSLE):")
    logger.info("=" * 100)
    logger.info(f"\n{results_df.to_string(index=False)}")
    logger.info("=" * 100)
    
    # Save results to CSV
    results_df.to_csv('model_comparison_results.csv', index=False)
    logger.info("Saved model comparison results to 'model_comparison_results.csv'")
    
    # Plot model comparison
    plt.figure(figsize=(14, 8))
    models = results_df['Model']
    rmsle_scores = results_df['RMSLE']
    
    # Sort bars by RMSLE score
    sorted_indices = np.argsort(rmsle_scores)
    sorted_models = [models.iloc[i] for i in sorted_indices]
    sorted_scores = [rmsle_scores.iloc[i] for i in sorted_indices]
    
    bar_plot = plt.bar(sorted_models, sorted_scores, color='skyblue')
    plt.xlabel('Models')
    plt.ylabel('RMSLE (Lower is Better)')
    plt.title('Model Comparison by RMSLE')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Annotate bars with RMSLE values
    for bar, score in zip(bar_plot, sorted_scores):
        plt.annotate(f'{score:.4f}', 
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('plots/model_comparison.png')
    plt.close()
    logger.info("Saved model comparison plot to 'plots/model_comparison.png'")
    
    # Recommendations based on results
    best_model_name = results_df.iloc[0]['Model']
    best_model_rmsle = results_df.iloc[0]['RMSLE']
    logger.info(f"\nBased on RMSLE, the best performing model is: {best_model_name} with RMSLE = {best_model_rmsle:.4f}")
    
    # Next steps for improvement
    logger.info("\nNext Steps for Model Improvement:")
    logger.info("1. Feature engineering - Create interaction terms and domain-specific features")
    logger.info("2. Hyperparameter tuning - Use GridSearchCV or RandomizedSearchCV for the best model")
    logger.info("3. Ensemble methods - Create a meta-model using predictions from top performers")
    logger.info("4. More advanced preprocessing - Try different scaling methods or outlier removal")
    
    logger.info("\nJob completed successfully!")
    
except Exception as e:
    logger.error(f"Error occurred: {str(e)}", exc_info=True)
    print(f"An error occurred: {str(e)}")