# data_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import seaborn as sns

# Assuming collect_from_database is a custom function to fetch data
def collect_from_database(query):
    # Mock implementation
    return pd.DataFrame()

# Data Collection
account_name = "rg_data_sci"
client_id = "a1b2c3d4"


dataset_from_database = collect_from_database("SELECT * FROM CLAIMS.DS_DATASET")

# Data Wrangling
total = dataset_from_database.isnull().sum()
percent = (dataset_from_database.isnull().sum() / dataset_from_database.isnull().count() * 100)

for i in dataset_from_database.columns:
    print(f"Column: {i}, dtype: {dataset_from_database[i].dtype}")

non_numerical = ['gender', 'marital_status', 'occupation', 'location', 'prev_claim_rejected', 'known_allergies']

# Encode categorical columns
label_encoders = {}
for column in non_numerical:
    label_encoders[column] = LabelEncoder()
    dataset_from_database[column] = label_encoders[column].fit_transform(dataset_from_database[column])

# Handle missing values
imputer = SimpleImputer(strategy='mean')
dataset_from_database = pd.DataFrame(imputer.fit_transform(dataset_from_database), columns=dataset_from_database.columns)

# Scale features
scaler = StandardScaler()
dataset_scaled = pd.DataFrame(scaler.fit_transform(dataset_from_database), columns=dataset_from_database.columns)

# Save preprocessed data
dataset_scaled.to_csv("preprocessed_data.csv", index=False)
