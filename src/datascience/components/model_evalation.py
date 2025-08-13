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
class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        os.environ['MLFLOW_TRACKING_URI']="https://dagshub.com/Vasim-rgb/end_end_ml.mlflow"
        os.environ['MLFLOW_TRACKING_USERNAME']="Vasim-rgb"
        os.environ["MLFLOW_TRACKING_PASSWORD"]="900abb7ecb44670bdc1d47d32ab703650e254e3f"
    def Evaluate(self):
        logger.info("evaluating the model")
       
        dagshub.init(repo_owner='vasim-rgb',
             repo_name='mlops-movie-recommender-end-to-end',
             mlflow=True)
        # Load similarity matrix
        sim = joblib.load( self.config.model_path)
        df = pd.read_csv(self.config.data_path)
     
        assert sim.shape[0] == len(movies_df), "Matrix rows and movie count mismatch!"
        mlflow.set_tracking_uri("https://dagshub.com/Vasim-rgb/mlops-movie-recommender-end-to-end.mlflow") 
        # Set your MLflow tracking URI
        mlflow.set_experiment("movie_recommendation_experiment")  # Set your experiment name
        # Ensure the artifacts directory exists
        
        
        # Start MLflow run
        mlflow.start_run(run_name="cosine_similarity_eval")

        # ---------------------
        # Quantitative Metrics
        # ---------------------
        shape = sim.shape
        mask = ~np.eye(shape[0], dtype=bool)  # ignore diagonal (self-similarity)
        values = sim[mask]

        metrics = {
            "matrix_rows": shape[0],
            "matrix_cols": shape[1],
            "avg_similarity": float(np.mean(values)),
            "max_similarity": float(np.max(values)),
            "min_similarity": float(np.min(values)),
            "std_similarity": float(np.std(values)),
            "sparsity": float(np.sum(values == 0) / values.size)
        }

        # Log metrics to MLflow
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # ---------------------
        # Qualitative Samples
        # ---------------------
        sample_movies = random.sample(range(len(movies_df)), min(5, len(movies_df)))
        qualitative_results = []

        for idx in sample_movies:
            title = movies_df.iloc[idx]['title']
            similar_indices = np.argsort(sim[idx])[::-1][1:6]  # top 5 excluding self
            recommended_titles = movies_df.iloc[similar_indices]['title'].tolist()

            qualitative_results.append({
                "movie": title,
                "recommendations": recommended_titles
            })

        # Save qualitative results as CSV for MLflow artifact
        os.makedirs(os.path.dirname(self.config.metric_file_name), exist_ok=True)
        qual_df = pd.DataFrame(qualitative_results)
        qual_path = self.config.metric_file_name
        qual_df.to_csv(qual_path, index=False)

        mlflow.log_artifact(qual_path)

        # ---------------------
        # Save similarity matrix as artifact
        # ---------------------
        joblib.dump(sim, "artifacts/sim.joblib")
        mlflow.log_artifact("artifacts/sim.joblib")

        mlflow.end_run()