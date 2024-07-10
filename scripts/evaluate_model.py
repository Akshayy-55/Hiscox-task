# evaluate_model.py

import pandas as pd
import xgboost as xgb
import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, cohen_kappa_score
import matplotlib.pyplot as plt

# Load preprocessed data
dataset = pd.read_csv("preprocessed_data.csv")

# Separate the DataFrame into labels and features
X = dataset.drop('claim_status', axis=1)
y = dataset['claim_status']

# Load the trained model
model = joblib.load("xgboost_model_optimized.pkl")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
cohen_kappa = cohen_kappa_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"ROC AUC Score: {roc_auc}")
print(f"Cohen Kappa Score: {cohen_kappa}")

# Feature Importance
fig1, ax1 = plt.subplots(figsize=(12, 6))
xgb.plot_importance(model, ax=ax1)
plt.savefig("feature_importance.png")

# ROC Curve
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig("roc_curve.png")
plt.show()
