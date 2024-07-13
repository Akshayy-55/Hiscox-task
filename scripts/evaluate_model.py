import pandas as pd
import xgboost as xgb
import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, cohen_kappa_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import logging
import os

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


def load_model(model_path):
    """
    Load the trained model from a file.

    Args:
    model_path (str): Path to the model file.

    Returns:
    Model: Loaded model.
    """
    try:
        logging.info(f"Loading model from {model_path}")
        model = joblib.load(model_path)
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and print evaluation metrics.

    Args:
    model: Trained model.
    X_test (pd.DataFrame): Test features.
    y_test (pd.Series): Test labels.

    Returns:
    dict: Dictionary containing evaluation metrics.
    """
    logging.info("Evaluating model")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
        "cohen_kappa": cohen_kappa_score(y_test, y_pred)
    }

    for metric, value in metrics.items():
        logging.info(f"{metric.capitalize()}: {value:.4f}")

    return metrics


def plot_feature_importance(model, output_path):
    """
    Plot and save feature importance.

    Args:
    model: Trained model.
    output_path (str): Path to save the feature importance plot.
    """
    logging.info("Plotting feature importance")
    fig, ax = plt.subplots(figsize=(12, 6))
    xgb.plot_importance(model, ax=ax)
    plt.savefig(output_path)
    logging.info(f"Feature importance plot saved to {output_path}")


def plot_roc_curve(y_test, y_pred_proba, output_path):
    """
    Plot and save ROC curve.

    Args:
    y_test (pd.Series): Test labels.
    y_pred_proba (np.array): Predicted probabilities for the positive class.
    output_path (str): Path to save the ROC curve plot.
    """
    logging.info("Plotting ROC curve")
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc_score(y_test, y_pred_proba):.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(output_path)
    logging.info(f"ROC curve plot saved to {output_path}")
    plt.show()


if __name__ == "__main__":
    data_path = os.getenv("DATA_PATH", "preprocessed_data.csv")
    model_path = os.getenv("MODEL_PATH", "xgboost_model_optimized.pkl")

    dataset = load_data(data_path)
    model = load_model(model_path)

    X = dataset.drop('claim_status', axis=1)
    y = dataset['claim_status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    metrics = evaluate_model(model, X_test, y_test)

    plot_feature_importance(model, "feature_importance.png")
    plot_roc_curve(y_test, model.predict_proba(X_test)[:, 1], "roc_curve.png")
