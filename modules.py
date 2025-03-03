import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification

class SyntheticDataLoader:
    def load(self):
        X, y = make_classification(n_samples=100, n_features=4, random_state=42)
        return {"X": X, "y": y}

class Normalizer:
    def process(self, data):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data["X"])
        return {"X": X_scaled, "y": data["y"]}

class LogisticModel:
    def __init__(self):
        self.model = LogisticRegression()

    def fit(self, data):
        self.model.fit(data["X"], data["y"])

    def predict(self, data):
        return self.model.predict(data["X"])

class AccuracyEvaluator:
    def evaluate(self, model, data):
        preds = model.predict(data)
        return accuracy_score(data["y"], preds)

import pandas as pd

class CSVDataLoader:
    def load(self, file):
        df = pd.read_csv(file)
        X = df.iloc[:, :-1].values  # All columns except last as features
        y = df.iloc[:, -1].values   # Last column as target
        return {"X": X, "y": y}
    
from sklearn.tree import DecisionTreeClassifier

class DecisionTreeModel:
    def __init__(self):
        self.model = DecisionTreeClassifier()

    def fit(self, data):
        self.model.fit(data["X"], data["y"])

    def predict(self, data):
        return self.model.predict(data["X"])