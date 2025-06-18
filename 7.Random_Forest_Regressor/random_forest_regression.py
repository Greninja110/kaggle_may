# In VS Code, save this as random_forest_regression.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(filename='random_forest_model.log', level=logging.INFO, 
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
    
    # Create a pipeline with preprocessing and random forest regression
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    # Train the model
    logger.info("Training random forest regression model...")
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
    
    # Get feature importances
    feature_names = model.named_steps['preprocessor'].get_feature_names_out()
    importances = model.named_steps['regressor'].feature_importances_
    
    # Log feature importances
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    importance_df = importance_df.sort_values('Importance', ascending=False)
    logger.info(f"Feature importances:\n{importance_df}")
    print("\nFeature importances:")
    print(importance_df)
    
    # Visualize actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Calories')
    plt.ylabel('Predicted Calories')
    plt.title('Random Forest Regression: Actual vs Predicted Calories')
    plt.savefig('random_forest_regression_results.png')
    plt.show()
    
    # Visualize feature importances
    plt.figure(figsize=(12, 8))
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Random Forest Regression: Feature Importances')
    plt.savefig('random_forest_feature_importances.png')
    plt.show()
    
except Exception as e:
    logger.error(f"Error occurred: {str(e)}", exc_info=True)
    print(f"An error occurred: {str(e)}")