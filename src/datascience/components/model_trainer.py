import pandas as pd
import os
from src.datascience import logger
import joblib
from src.datascience.entity.config_entity import ModelTrainerConfig
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pickle
class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        logger.info("Training the model")
        df = pd.read_csv(self.config.train_data_path)
        # Use the correct column name and drop NaNs
        df = df.dropna(subset=['data', 'target'])
        tfi = TfidfVectorizer(max_features=5000)
        x = tfi.fit_transform(df['data']).toarray()
        y = df['target'].values
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        mnb = MultinomialNB()
        mnb.fit(x_train, y_train)
        # save the vectorizer
        with open(os.path.join(self.config.root_dir, 'vectorizer.pkl'), 'wb') as f:
            pickle.dump(tfi, f)
        #stack xtrain y train and save it
        with open(os.path.join(self.config.root_dir, 'train_test.pkl'), 'wb') as f:
            pickle.dump((x_train, y_train, x_test, y_test), f)
        # Save model
        joblib.dump(mnb, os.path.join(self.config.root_dir, self.config.model_name))
        logger.info(f"Model trained and saved at {self.config.root_dir}/{self.config.model_name}")



