from flask import Flask, render_template, request
import pickle

from src.datascience.pipeline.prediction_pipeline import PredictionPipeline

app = Flask(__name__)
prediction_pipeline = PredictionPipeline()


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    user_input = ""
    
    if request.method == "POST":
        user_input = request.form["message"]
        preprocessed_input = prediction_pipeline.preprocess(user_input)
        result, _ = prediction_pipeline.predict(preprocessed_input)
        prediction = "🚫 Spam Message" if result == 1 else "✅ Not Spam"
    
    return render_template("index.html", prediction=prediction, user_input=user_input)

if __name__ == "__main__":
    app.run(debug=True)