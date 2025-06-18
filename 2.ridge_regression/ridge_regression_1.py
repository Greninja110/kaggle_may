# In VS Code, save this as ridge_regression.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import logging
import traceback

# Set up logging
logging.basicConfig(filename='ridge_regression.log', level=logging.INFO, 
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
    
    # Preprocessing for categorical variable (Sex)
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first'), ['Sex'])
        ],
        remainder='passthrough'
    )
    
    # Create a pipeline with preprocessing and Ridge regression
    # Alpha parameter controls the strength of regularization
    alpha_value = 1.0  # You can tune this hyperparameter
    logger.info(f"Using Ridge Regression with alpha={alpha_value}")
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', Ridge(alpha=alpha_value))
    ])
    
    # Train the model
    logger.info("Training Ridge regression model...")
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
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
    
    # Get feature names after preprocessing (for coefficient analysis)
    # This extracts feature names after one-hot encoding
    cat_features = ['Sex_Male']  # The OneHotEncoder with drop='first' creates this feature
    num_features = X.drop(['Sex'], axis=1).columns.tolist()
    feature_names = cat_features + num_features
    
    # Extract and log coefficients (only works when the pipeline is fitted)
    try:
        coefficients = model.named_steps['regressor'].coef_
        intercept = model.named_steps['regressor'].intercept_
        
        logger.info("Model coefficients:")
        logger.info(f"Intercept: {intercept:.4f}")
        
        for feature, coef in zip(feature_names, coefficients):
            logger.info(f"{feature}: {coef:.4f}")
            
        print("Model coefficients:")
        print(f"Intercept: {intercept:.4f}")
        for feature, coef in zip(feature_names, coefficients):
            print(f"{feature}: {coef:.4f}")
    except:
        logger.warning("Could not extract coefficients", exc_info=True)
    
    # Visualize actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Calories')
    plt.ylabel('Predicted Calories')
    plt.title('Ridge Regression (alpha={:.2f}): Actual vs Predicted Calories'.format(alpha_value))
    plt.savefig('ridge_regression_results.png')
    plt.show()
    
except Exception as e:
    logger.error(f"Error occurred: {str(e)}")
    logger.error(traceback.format_exc())
    print(f"An error occurred: {str(e)}")
    print("See ridge_regression.log for details")