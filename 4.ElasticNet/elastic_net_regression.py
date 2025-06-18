# In VS Code, save this as elastic_net_regression.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(filename='elastic_net_model.log', level=logging.INFO, 
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
    
    # Create a pipeline with preprocessing and ElasticNet regression
    # alpha controls the regularization strength
    # l1_ratio controls the balance between L1 and L2 (1 = Lasso, 0 = Ridge)
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000, random_state=42))
    ])
    
    # Train the model
    logger.info("Training ElasticNet regression model...")
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
    
    # Get feature importance (based on coefficients)
    # First, get the feature names after one-hot encoding
    cat_feature = ['Sex_male']  # Since we used drop='first', only male is encoded
    num_features = X.drop(['Sex'], axis=1).columns.tolist()
    all_features = cat_feature + num_features
    
    # Get the coefficients from the ElasticNet model
    coefficients = model.named_steps['regressor'].coef_
    
    # Create a DataFrame of feature importances
    feature_importance = pd.DataFrame({'Feature': all_features, 'Coefficient': coefficients})
    feature_importance = feature_importance.sort_values(by='Coefficient', key=abs, ascending=False)
    
    logger.info("Feature importance (coefficients):")
    logger.info(feature_importance)
    print("\nFeature importance (coefficients):")
    print(feature_importance)
    
    # Visualize actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Calories')
    plt.ylabel('Predicted Calories')
    plt.title('ElasticNet Regression: Actual vs Predicted Calories')
    plt.savefig('elastic_net_regression_results.png')
    
    # Visualize feature importance
    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance['Feature'], feature_importance['Coefficient'])
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature')
    plt.title('ElasticNet Regression: Feature Importance')
    plt.tight_layout()
    plt.savefig('elastic_net_feature_importance.png')
    plt.show()
    
except Exception as e:
    logger.error(f"Error occurred: {str(e)}", exc_info=True)
    print(f"An error occurred: {str(e)}")