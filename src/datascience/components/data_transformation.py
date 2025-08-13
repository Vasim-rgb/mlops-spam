import os
from src.datascience import logger
from src.datascience.entity.config_entity import DataTransformationConfig
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.stem.porter import PorterStemmer
import string
from src.datascience.utils.transformation_fn import transform

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    
    ## Note: You can add different data transformation techniques such as Scaler, PCA and all
    
    def train_test_splitting(self):
        try:
            txt = pd.read_csv(self.config.data_path, encoding='latin1')
            l=LabelEncoder()
            txt['target']=l.fit_transform(txt['target'])
            if(nltk.download('punkt') and nltk.download('stopwords') and  nltk.download('punkt_tab')):
                logger.info("nltk data downloaded successfully")
                txt['num_words']=txt['text'].apply(lambda x:len(nltk.word_tokenize(x)))
                logger.info("Number of words in each text column calculated")
                txt['num_chars']=txt['text'].apply(lambda x:len(x))
                logger.info("Number of characters in each text column calculated")
                plt.figure(figsize=(12,6))
                logger.info("Plotting the distribution of number of words and characters in each text column")
                sns.histplot(txt['num_chars'], label='num_chars', color='blue', alpha=0.5)
                sns.histplot(txt['num_words'], label='num_words', color='orange', alpha=0.5)
                plt.legend()
                plot_path = os.path.join(self.config.root_dir, "word_char_distribution.png")
                plt.savefig(plot_path)
                plt.close()
                logger.info(f"Plot saved at: {plot_path}")
                txt['text']=txt['text'].apply(transform)
                logger.info("Text column transformed using nltk")
                txt['newt']=txt['text'].apply(lambda x: ' '.join(x))
                logger.info("New text column created by joining the transformed text")
                txt['newt']=txt['text'].apply(lambda x: ' '.join(x))
                df=pd.DataFrame({'target':txt['target'],'data':txt['newt']})
                logger.info("Dataframe created with target and new text column")
                df.to_csv(self.config.transformed_data_path, index=False)
                logger.info("Transformed data saved at: {}".format(self.config.transformed_data_path))
            else:
                logger.error("nltk data download failed")
                raise Exception("nltk data download failed")
        except Exception as e:
            logger.exception(f"Error occurred during data transformation: {e}")
            raise e


        logger.info("transofmation completed and data saved at: {}".format(self.config.transformed_data_path))
        