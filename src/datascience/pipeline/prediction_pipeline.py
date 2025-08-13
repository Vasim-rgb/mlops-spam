import os
import joblib
import pickle
import numpy as np
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download necessary NLTK data (only first run)
nltk.download('punkt')
nltk.download('stopwords')

class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load(r'artifacts/model_trainer/model.joblib')

        with open(r'artifacts/model_trainer/vectorizer.pkl', "rb") as f:
            self.vectorizer = pickle.load(f)

        self.ps = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

    def preprocess(self, text):
        # Lowercase
        text = text.lower()
        # Tokenize
        tokens = word_tokenize(text)
        # Keep only alphanumeric
        tokens = [t for t in tokens if t.isalnum()]
        # Remove stopwords and punctuation
        tokens = [t for t in tokens if t not in self.stop_words and t not in string.punctuation]
        # Stemming
        tokens = [self.ps.stem(t) for t in tokens]
        # Return joined string
        return ' '.join(tokens)

    def predict(self, texts):
        if isinstance(texts, str):
            texts = [texts]  # make it a list

        # Preprocess all texts
        processed_texts = [self.preprocess(t) for t in texts]

        # Transform using vectorizer
        features = self.vectorizer.transform(processed_texts)

        # Predictions
        predictions = self.model.predict(features)
        probabilities = self.model.predict_proba(features)

        return predictions, probabilities


if __name__ == "__main__":
    pipeline = PredictionPipeline()

    sample_texts = [
        "Congratulations! 🎉 You have won a $1000 Walmart gift card. Click here to claim your prize: http://bit.ly/fake-offer.",
        "Hi, are we still meeting for lunch today?"
    ]

    preds, probs = pipeline.predict(sample_texts)

    for text, label, prob in zip(sample_texts, preds, probs):
        print(f"Text: {text}")
        print(f"Predicted Label: {label}")
        print(f"Probabilities: {prob}")
        if label == 1:
            print("This is a spam message.")
        else:
            print("This is not spam.")
        print("-" * 40)