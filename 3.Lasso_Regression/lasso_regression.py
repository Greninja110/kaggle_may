# In VS Code, save this as lasso_regression.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(filename='lasso_regression.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

try:
    # Load the data
    logger.info("Loading data...")
    data = pd.read_csv('playground-series-s5e5/train.csv')
    
    # Display basic info
    logger.info(f"Dataset shape: {data.shape}")
    logger.info(f"Dataset columns: {data.columns.tolist()}")
    logger.info(f"Dataset info:\n{data.dtypes}")
    
    # Preparing features and target
    X = data.drop(['id', 'Calories'], axis=1)
    y = data['Calories']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info(f"Training set size: {X_train.shape}, Test set size: {X_test.shape}")
    
    # Preprocessing for categorical variable (Sex) and scaling numerical features
    # Lasso benefits from feature scaling, so we add StandardScaler
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first'), ['Sex']),
            ('num', StandardScaler(), X.select_dtypes(include=['int64', 'float64']).columns.drop('Sex', errors='ignore'))
        ],
        remainder='passthrough'
    )
    
    # Create a pipeline with preprocessing and Lasso regression
    # We'll use GridSearchCV to find the optimal alpha parameter
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', Lasso(random_state=42, max_iter=10000))
    ])
    
    # Define parameters for grid search
    param_grid = {
    'regressor__alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    }
    
    # Grid search with cross-validation
    logger.info("Performing grid search for optimal alpha parameter...")
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Get best parameters and model
    best_alpha = grid_search.best_params_['regressor__alpha']
    best_model = grid_search.best_estimator_
    logger.info(f"Best alpha value: {best_alpha}")
    
    # Make predictions with best model
    y_pred = best_model.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    logger.info(f"Model performance:")
    logger.info(f"Mean Squared Error: {mse:.2f}")
    logger.info(f"Root Mean Squared Error: {rmse:.2f}")
    logger.info(f"R² Score: {r2:.2f}")
    
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")
    
    # Get feature importance (coefficient values)
    # First get the column names after preprocessing
    categorical_feature = ['Sex']
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.drop('Sex', errors='ignore').tolist()
    
    # Get coefficients and feature names
    lasso_model = best_model.named_steps['regressor']
    coefficients = lasso_model.coef_
    
    # Create a dataframe of features and their coefficients
    # Note: This will depend on how the preprocessor transforms the features
    # For one-hot encoding, we need to handle the transformed feature names
    logger.info("Lasso coefficient values (feature importance):")
    logger.info(f"Intercept: {lasso_model.intercept_:.4f}")
    
    if hasattr(best_model.named_steps['preprocessor'], 'transformers_'):
        # Try to get feature names from preprocessor
        try:
            # This part can be tricky as the feature names might not be accessible directly
            logger.info("Feature coefficients may not be fully interpretable due to preprocessing transformations")
            for i, coef in enumerate(coefficients):
                if abs(coef) > 0:  # Only log non-zero coefficients
                    logger.info(f"Feature {i}: {coef:.4f}")
        except Exception as e:
            logger.warning(f"Could not map coefficients to feature names: {str(e)}")
    
    # Visualize actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Calories')
    plt.ylabel('Predicted Calories')
    plt.title('Lasso Regression: Actual vs Predicted Calories')
    plt.savefig('lasso_regression_results.png')
    plt.show()
    
    # Visualize non-zero coefficients
    non_zero_coefs = [c for c in coefficients if c != 0]
    if len(non_zero_coefs) > 0:
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(non_zero_coefs)), non_zero_coefs)
        plt.xlabel('Feature Index')
        plt.ylabel('Coefficient Value')
        plt.title('Lasso Regression: Non-Zero Coefficients')
        plt.savefig('lasso_coefficients.png')
        plt.show()
    
    logger.info(f"Number of features used: {np.count_nonzero(coefficients)} out of {len(coefficients)}")
    print(f"Number of features used: {np.count_nonzero(coefficients)} out of {len(coefficients)}")
    
except Exception as e:
    logger.error(f"Error occurred: {str(e)}", exc_info=True)
    print(f"An error occurred: {str(e)}")