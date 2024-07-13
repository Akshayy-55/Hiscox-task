import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import joblib
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_data(file_path):
    """
    Load preprocessed data from a CSV file.

    Args:
    file_path (str): Path to the preprocessed data file.

    Returns:
    pd.DataFrame: DataFrame containing the preprocessed data.
    """
    try:
        logging.info(f"Loading data from {file_path}")
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise


def train_model(X, y):
    """
    Train an XGBoost model using the provided features and labels.

    Args:
    X (pd.DataFrame): Features for training.
    y (pd.Series): Labels for training.

    Returns:
    model: Trained XGBoost model.
    """
    logging.info("Training model")
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X, y)
    return model


def save_model(model, model_path):
    """
    Save the trained model to a file.

    Args:
    model: Trained model.
    model_path (str): Path to save the model.
    """
    joblib.dump(model, model_path)
    logging.info(f"Model saved to {model_path}")


if __name__ == "__main__":
    data_path = "preprocessed_data.csv"
    model_path = "xgboost_model_optimized.pkl"

    dataset = load_data(data_path)

    X = dataset.drop('claim_status', axis=1)
    y = dataset['claim_status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = train_model(X_train, y_train)
    save_model(model, model_path)
