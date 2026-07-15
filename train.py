import json
import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# 1. Load hyperparameters from config file
with open("config.json", "r") as f:
    config = json.load(f)

# 2. Generate synthetic dataset for the experiment
X, y = make_classification(
    n_samples=1000, 
    n_features=10, 
    n_informative=8, 
    random_state=config["random_state"]
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    size=0.2, 
    random_state=config["random_state"]
)

# 3. Initialize and train the selected model
if config["model_type"] == "RandomForest":
    model = RandomForestClassifier(
        n_estimators=config["n_estimators"], 
        max_depth=config["max_depth"], 
        random_state=config["random_state"]
    )
elif config["model_type"] == "GradientBoosting":
    model = GradientBoostingClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        learning_rate=config.get("learning_rate", 0.1),
        random_state=config["random_state"]
    )
else:
    raise ValueError(f"Unsupported model type: {config['model_type']}")

model.fit(X_train, y_train)

# 4. Evaluate model performance
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# 5. Save evaluation metrics and parameters
metrics = {
    "model_type": config["model_type"],
    "n_estimators": config["n_estimators"],
    "max_depth": config["max_depth"],
    "accuracy": round(acc, 4)
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print(f"✅ Training completed. New accuracy: {acc:.4f}")