from src.datascience.config.configuration import ConfigurationManager
from src.datascience.components.model_evalation import ModelEvaluation
from src.datascience import logger

STAGE_NAME = "Model evaluation stage"

class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass

    def initiate_model_evaluation(self):
        logger.info(f"{STAGE_NAME} started")
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(config=model_evaluation_config)
        model_evaluation.Evaluate()
        logger.info(f"{STAGE_NAME} completed successfully.")
        
if __name__ == "__main__":
    try:
        logger.info(">>>>> stage started <<<<<")    
        model_evaluation_pipeline = ModelEvaluationTrainingPipeline()
        model_evaluation_pipeline.initiate_model_evaluation()
        logger.info(">>>>> stage completed <<<<<")
    except Exception as e:
        logger.exception(e)
        raise e