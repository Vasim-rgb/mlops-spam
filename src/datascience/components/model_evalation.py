import pandas as pd
import os
from src.datascience import logger
import joblib
import numpy as np
import mlflow
import joblib
import random
import os,dagshub
from src.datascience.entity.config_entity import ModelEvaluationConfig
import pickle
from sklearn.metrics import classification_report
import json
class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        os.environ['MLFLOW_TRACKING_URI']="https://dagshub.com/Vasim-rgb/mlops-spam.mlflow"
        os.environ['MLFLOW_TRACKING_USERNAME']="Vasim-rgb"
        os.environ["MLFLOW_TRACKING_PASSWORD"]="900abb7ecb44670bdc1d47d32ab703650e254e3f"
    def Evaluate(self):
        logger.info("evaluating the model")
       
        dagshub.init(
        repo_owner='vasim-rgb',
        repo_name='mlops-spam',
        mlflow=True
    )

        # Load model
        mnb = joblib.load(self.config.model_path)
        with open(self.config.test_data_path, 'rb') as f:
            x_train, y_train, x_test, y_test = pickle.load(f)

        # Predict on test data
        y_pred = mnb.predict(x_test)
        accuracy = np.mean(y_pred == y_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        logger.info(f"Model accuracy: {accuracy}")

        mlflow.set_tracking_uri("https://dagshub.com/Vasim-rgb/mlops-spam.mlflow")
        mlflow.set_experiment("mlops-spam-experiment")
        with mlflow.start_run():
            mlflow.log_param("model_name", type(mnb).__name__)
            mlflow.log_metric("accuracy", accuracy)
            # Log classification report metrics
            for label, scores in report.items():
                if isinstance(scores, dict):
                    for metric_name, value in scores.items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(f"{label}_{metric_name}", value)
            # Save classification report as artifact
            os.makedirs(os.path.dirname(self.config.metric_file_name), exist_ok=True)
            with open(self.config.metric_file_name, "w") as f:
                json.dump(report, f)
            mlflow.log_artifact(self.config.metric_file_name)
            # Save and log model as artifact (not registry)
            joblib.dump(mnb, "model.joblib")
            mlflow.log_artifact("model.joblib")