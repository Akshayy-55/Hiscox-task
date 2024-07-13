import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def collect_from_database(query):
    """
    Fetch data from the database using the provided query.

    Args:
    query (str): SQL query to fetch data.

    Returns:
    pd.DataFrame: DataFrame containing the fetched data.
    """
    # Actual implementation to fetch data from the database
    # Example:
    # import pyodbc
    # connection = pyodbc.connect('DRIVER={SQL Server};SERVER=server_name;DATABASE=db_name;UID=user;PWD=password')
    # return pd.read_sql(query, connection)
    logging.info("Fetching data from database")
    return pd.DataFrame()  # Replace with actual data fetching logic


def preprocess_data(data):
    """
    Preprocess the data by encoding categorical columns, handling missing values, and scaling features.

    Args:
    data (pd.DataFrame): DataFrame containing the raw data.

    Returns:
    pd.DataFrame: DataFrame containing the preprocessed data.
    """
    logging.info("Starting preprocessing")

    # Print column names and data types
    for col in data.columns:
        logging.info(f"Column: {col}, dtype: {data[col].dtype}")

    # Encode categorical columns
    non_numerical = ['gender', 'marital_status', 'occupation', 'location', 'prev_claim_rejected', 'known_allergies']
    label_encoders = {}
    for column in non_numerical:
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column].astype(str))

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

    # Scale features
    scaler = StandardScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    logging.info("Preprocessing complete")
    return data_scaled


def save_preprocessed_data(data, filename):
    """
    Save the preprocessed data to a CSV file.

    Args:
    data (pd.DataFrame): DataFrame containing the preprocessed data.
    filename (str): Name of the file to save the data.
    """
    data.to_csv(filename, index=False)
    logging.info(f"Preprocessed data saved to {filename}")


if __name__ == "__main__":
    account_name = os.getenv("ACCOUNT_NAME", "rg_data_sci")
    client_id = os.getenv("CLIENT_ID", "a1b2c3d4")

    dataset_from_database = collect_from_database("SELECT * FROM CLAIMS.DS_DATASET")

    if not dataset_from_database.empty:
        preprocessed_data = preprocess_data(dataset_from_database)
        save_preprocessed_data(preprocessed_data, "preprocessed_data.csv")
    else:
        logging.error("No data fetched from the database")
