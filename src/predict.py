import os
import joblib
import yaml
import pandas as pd
from sklearn.metrics import (precision_score, recall_score, 
                             f1_score, classification_report)


class Predictor:
    def __init__(self):
        with open('config.yml', 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)

        self.model_path = config['model']['store_path']
        self.pred_threshold = config['model']['threshold']

        model_file_path = os.path.join(self.model_path, 'model.pkl')

        self.model = joblib.load(model_file_path)
    

    def evaluate_model(self, test_data: pd.DataFrame):
        """Evaluates model and computes its precision, recall and pr-auc"""
        X_test, y_test = test_data['text'].values, test_data['label'].values

        # y_pred = self.model.predict(X_test)
        y_pred = (self.model.predict_proba(X_test)[:, 1] > self.pred_threshold) * 1

        return (
            precision_score(y_test, y_pred),
            recall_score(y_test, y_pred),
            f1_score(y_test, y_pred, average='macro'),
            classification_report(y_test, y_pred)
        )
