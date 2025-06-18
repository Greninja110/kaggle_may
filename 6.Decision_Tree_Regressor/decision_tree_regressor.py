# In VS Code, save this as decision_tree_regressor.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(filename='decision_tree_model.log', level=logging.INFO, 
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
    
    # Create a pipeline with preprocessing and decision tree regressor
    # Using max_depth to prevent overfitting and random_state for reproducibility
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', DecisionTreeRegressor(max_depth=5, random_state=42))
    ])
    
    # Train the model
    logger.info("Training decision tree regressor model...")
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
    plt.title('Decision Tree Regressor: Actual vs Predicted Calories')
    plt.savefig('decision_tree_results.png')
    plt.show()
    
    # Additional visualizations specific to Decision Trees
    if hasattr(model['regressor'], 'feature_importances_'):
        try:
            # Get feature names after transformation
            cat_features = model['preprocessor'].transformers_[0][1].get_feature_names_out(['Sex'])
            num_features = X.columns.drop('Sex')
            feature_names = np.concatenate([cat_features, num_features])
        except AttributeError:
            # Fallback for older scikit-learn versions
            logger.warning("Could not get transformed feature names. Using generic feature names.")
            n_cat_features = 1  # We're using drop='first' for one-hot encoding
            feature_names = np.array([f"Sex_encoded"] + list(X.columns.drop('Sex')))
        
        # Get feature importances
        importances = model['regressor'].feature_importances_
        
        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]
        
        # Plot feature importances
        plt.figure(figsize=(12, 6))
        plt.title('Feature Importances')
        plt.bar(range(len(importances)), importances[indices], align='center')
        plt.xticks(range(len(importances)), feature_names[indices], rotation=90)
        plt.tight_layout()
        plt.savefig('decision_tree_feature_importances.png')
        plt.show()
        
        # Log feature importances
        logger.info("Feature importances:")
        for i in range(len(importances)):
            logger.info(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
    
except Exception as e:
    logger.error(f"Error occurred: {str(e)}", exc_info=True)
    print(f"An error occurred: {str(e)}")