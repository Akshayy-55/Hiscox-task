# train_model.py

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score

# Load preprocessed data
dataset = pd.read_csv("preprocessed_data.csv")

# Separate the DataFrame into labels and features
X = dataset.drop('claim_status', axis=1)
y = dataset['claim_status']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initial Model Training
model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False
)

model.fit(X_train, y_train)

# Cross Validation
param_dist = {
    'n_estimators': range(50, 400, 50),
    'learning_rate': [0.01, 0.05, 0.1, 0.3],
    'max_depth': range(3, 10, 2),
    'min_child_weight': [1, 3, 5, 7],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

random_search = RandomizedSearchCV(
    estimator=xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False),
    param_distributions=param_dist,
    scoring=make_scorer(f1_score),
    n_iter=50,
    cv=3,
    verbose=1,
    random_state=42
)

random_search.fit(X_train, y_train)
model_optimized = random_search.best_estimator_


# Save the model
import joblib
joblib.dump(model_optimized, "xgboost_model_optimized.pkl")
