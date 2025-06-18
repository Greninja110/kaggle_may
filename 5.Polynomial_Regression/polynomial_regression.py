# In VS Code, save this as polynomial_regression.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(filename='polynomial_regression.log', level=logging.INFO, 
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
    categorical_preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first'), ['Sex'])
        ],
        remainder='passthrough'
    )
    
    # Create a pipeline with preprocessing, polynomial features, and linear regression
    model = Pipeline([
        ('preprocessor', categorical_preprocessor),
        ('polynomial', PolynomialFeatures(degree=2)),  # Add polynomial features with degree 2
        ('regressor', LinearRegression())
    ])
    
    # Train the model
    logger.info("Training polynomial regression model...")
    logger.info("Using polynomial features with degree=2")
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
    
    # Visualize actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Calories')
    plt.ylabel('Predicted Calories')
    plt.title('Polynomial Regression: Actual vs Predicted Calories')
    plt.savefig('polynomial_regression_results.png')
    plt.show()
    
    # Additional debugging - coefficients analysis
    if hasattr(model['regressor'], 'coef_'):
        logger.info("Model coefficients analysis:")
        poly_features = model['polynomial'].get_feature_names_out(
            model['preprocessor'].get_feature_names_out()
        )
        coefficients = list(zip(poly_features, model['regressor'].coef_))
        logger.info(f"Top 10 most important features: {sorted(coefficients, key=lambda x: abs(x[1]), reverse=True)[:10]}")
    
except Exception as e:
    logger.error(f"Error occurred: {str(e)}", exc_info=True)
    print(f"An error occurred: {str(e)}")