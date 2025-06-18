# In VS Code, save this as feature_engineering_fix.py
import joblib
import pandas as pd

# Define named functions instead of lambda functions
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

# Store function references in a dictionary 
feature_engineering_steps = {
    'create_BMI': create_BMI,
    'create_Exercise_Intensity': create_Exercise_Intensity,
    'create_Thermal_Load': create_Thermal_Load,
    'create_Work_Load': create_Work_Load,
    'create_Heart_Rate_Reserve': create_Heart_Rate_Reserve,
    'create_MET_Approx': create_MET_Approx,
    'create_Energy_Score': create_Energy_Score
}

# Save the feature engineering steps
joblib.dump(feature_engineering_steps, 'models/feature_engineering_steps.pkl')

print("Feature engineering functions saved successfully!")