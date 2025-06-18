# In VS Code, save this as nn_calorie_predictor.py
import os
import time
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, TerminateOnNaN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import logging
import psutil
import GPUtil
from datetime import datetime

# Configure TensorFlow to use GPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"Using GPU: {tf.test.gpu_device_name()}")
else:
    print("No GPU found, using CPU instead")

# Set up logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"nn_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(filename=log_file, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Create a console handler to display logs in terminal
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

# Fixed RMSLE function to prevent memory errors and handle NaN values
def rmsle(y_true, y_pred):
    """
    Calculate Root Mean Squared Logarithmic Error
    RMSLE = sqrt(1/n * sum((log(1+y_pred) - log(1+y_true))^2))
    """
    # Convert pandas Series to numpy arrays to avoid memory issues
    if hasattr(y_true, 'values'):
        y_true = y_true.values
    if hasattr(y_pred, 'values'):
        y_pred = y_pred.values
    
    # Ensure predictions are non-negative (required for log)
    y_pred = np.maximum(0, y_pred)
    
    # Calculate RMSLE element-wise to avoid large intermediate arrays
    log_diff = np.log1p(y_pred) - np.log1p(y_true)
    return np.sqrt(np.mean(np.square(log_diff)))

# Fixed Keras metric for RMSLE
def rmsle_keras(y_true, y_pred):
    # Ensure predictions are non-negative
    y_pred = tf.maximum(y_pred, 0.0)
    return tf.sqrt(tf.reduce_mean(tf.square(tf.math.log1p(y_pred) - tf.math.log1p(y_true))))

# Function to get system stats
def get_system_stats():
    cpu_percent = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()
    memory_percent = memory_info.percent
    
    gpu_stats = {}
    try:
        gpus = GPUtil.getGPUs()
        for i, gpu in enumerate(gpus):
            gpu_stats[f"GPU-{i}"] = {
                "name": gpu.name,
                "load": f"{gpu.load*100:.1f}%",
                "memory_used": f"{gpu.memoryUsed}MB/{gpu.memoryTotal}MB ({gpu.memoryUtil*100:.1f}%)"
            }
    except Exception as e:
        gpu_stats["Error"] = str(e)
    
    return {
        "CPU Usage": f"{cpu_percent}%",
        "Memory Usage": f"{memory_percent}% ({memory_info.used/(1024**3):.1f}GB/{memory_info.total/(1024**3):.1f}GB)",
        "GPU Stats": gpu_stats
    }

try:
    start_time = time.time()
    logger.info("Starting neural network training for calorie prediction")
    
    # Log system info
    logger.info(f"System Info: Python {os.sys.version}")
    logger.info(f"TensorFlow version: {tf.__version__}")
    logger.info(f"Initial system stats: {get_system_stats()}")
    
    # Load the data
    logger.info("Loading data...")
    data_dir = "playground-series-s5e5"
    train_data = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    test_data = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    
    # Display basic info
    logger.info(f"Training dataset shape: {train_data.shape}")
    logger.info(f"Test dataset shape: {test_data.shape}")
    logger.info(f"Training dataset columns: {train_data.columns.tolist()}")
    
    # Preparing features and target
    X = train_data.drop(['id', 'Calories'], axis=1)
    y = train_data['Calories']
    
    # Save test IDs for submission
    test_ids = test_data['id']
    X_test = test_data.drop(['id'], axis=1)
    
    # Define categorical and numerical columns
    categorical_cols = ['Sex']
    numerical_cols = [col for col in X.columns if col not in categorical_cols]
    
    # Create preprocessing pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first'), categorical_cols)
        ]
    )
    
    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    logger.info(f"Training set size: {X_train.shape}, Validation set size: {X_val.shape}")
    
    # Fit the preprocessor and transform the data
    logger.info("Preprocessing data...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get the number of features after preprocessing
    n_features = X_train_processed.shape[1]
    logger.info(f"Number of features after preprocessing: {n_features}")
    
    # Build the neural network model with regularization
    logger.info("Building neural network model...")
    model = Sequential([
        Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001), input_shape=(n_features,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        BatchNormalization(),
        Dense(1, activation='relu')  # Using ReLU to ensure non-negative predictions
    ])
    
    # Compile the model with RMSLE as a metric
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',  # Using MSE as loss function
        metrics=[rmsle_keras]
    )
    
    # Print model summary to logger
    model.summary(print_fn=lambda x: logger.info(x))
    
    # Set up callbacks with TerminateOnNaN added
    callbacks = [
        EarlyStopping(
            monitor='val_rmsle_keras',
            patience=20,
            restore_best_weights=True,
            mode='min'
        ),
        ModelCheckpoint(
            filepath='best_model.h5',
            monitor='val_rmsle_keras',
            save_best_only=True,
            mode='min'
        ),
        TensorBoard(log_dir='./logs/tensorboard'),
        TerminateOnNaN()  # Terminate training if NaN values are encountered
    ]
    
    # Train the model with increased batch size
    logger.info("Training neural network...")
    training_start = time.time()
    
    history = model.fit(
        X_train_processed, y_train,
        validation_data=(X_val_processed, y_val),
        epochs=100,
        batch_size=64,  # Increased batch size to reduce memory usage
        callbacks=callbacks,
        verbose=1
    )
    
    training_duration = time.time() - training_start
    logger.info(f"Model training completed in {training_duration:.2f} seconds")
    logger.info(f"System stats after training: {get_system_stats()}")
    
    # Evaluate the model on validation set
    logger.info("Evaluating model on validation set...")
    y_val_pred = model.predict(X_val_processed)
    
    # Ensure predictions are non-negative before RMSLE calculation
    y_val_pred = np.maximum(0, y_val_pred)
    
    # Calculate RMSLE manually with fixed function
    val_rmsle = rmsle(y_val, y_val_pred)
    logger.info(f"Validation RMSLE: {val_rmsle:.4f}")
    
    # Make predictions on test set
    logger.info("Making predictions on test set...")
    test_predictions = model.predict(X_test_processed)
    
    # Ensure test predictions are non-negative
    test_predictions = np.maximum(0, test_predictions)
    
    # Create submission file
    submission = pd.DataFrame({
        'id': test_ids,
        'Calories': test_predictions.flatten()
    })
    submission.to_csv('nn_submission.csv', index=False)
    logger.info("Submission file created: nn_submission.csv")
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Handle possible NaN values in history
    if 'rmsle_keras' in history.history and 'val_rmsle_keras' in history.history:
        # Replace NaN values with None for plotting
        rmsle_train = [x if not np.isnan(x) else None for x in history.history['rmsle_keras']]
        rmsle_val = [x if not np.isnan(x) else None for x in history.history['val_rmsle_keras']]
        
        plt.subplot(1, 2, 2)
        plt.plot(rmsle_train, label='Training RMSLE')
        plt.plot(rmsle_val, label='Validation RMSLE')
        plt.title('Model RMSLE')
        plt.xlabel('Epoch')
        plt.ylabel('RMSLE')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    logger.info("Training history plot saved: training_history.png")
    
    # Plot actual vs predicted values
    y_val_true = y_val.values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_val_true, y_val_pred, alpha=0.7)
    plt.plot([y_val_true.min(), y_val_true.max()], [y_val_true.min(), y_val_true.max()], 'r--')
    plt.xlabel('Actual Calories')
    plt.ylabel('Predicted Calories')
    plt.title('Neural Network: Actual vs Predicted Calories')
    plt.savefig('nn_predictions.png')
    logger.info("Predictions plot saved: nn_predictions.png")
    
    # Performance summary
    total_duration = time.time() - start_time
    logger.info("\n" + "="*50)
    logger.info("PERFORMANCE SUMMARY")
    logger.info("="*50)
    logger.info(f"Total execution time: {total_duration:.2f} seconds")
    logger.info(f"Training time: {training_duration:.2f} seconds")
    logger.info(f"Final validation RMSLE: {val_rmsle:.4f}")
    logger.info("="*50)
    
except Exception as e:
    logger.error(f"Error occurred: {str(e)}", exc_info=True)
    print(f"An error occurred: {str(e)}")