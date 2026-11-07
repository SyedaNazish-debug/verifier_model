
from flask import Flask, render_template, request
import joblib
import os

from text_preprocessing import clean_text

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# LOAD EXISTING TRAINED MODEL AND VECTORIZER
model = joblib.load(
    os.path.join(BASE_DIR, "model", "VERIFIER LENSE MODEL.pkl")
)

vectorizer = joblib.load(
    os.path.join(BASE_DIR, "model", "TF-IDF_VECTORIZIER.pkl")
)


# DASHBOARD
@app.route("/")
def home():
    return render_template("index.html")


# NEWS ANALYZER
@app.route("/analyzer")
def analyzer():
    return render_template("analyzer.html")


# MODEL INSIGHTS
@app.route("/model-insights")
def model_insights():
    return render_template("model_insights.html")


# LEARN AND UNDERSTAND
@app.route("/learn")
def learn():
    return render_template("learn.html")


# ABOUT THE PROJECT
@app.route("/about")
def about():
    return render_template("about.html")


# NEWS PREDICTION
@app.route("/predict", methods=["POST"])
def predict():

    news_text = request.form.get("news_text", "").strip()

    # Validate user input
    if not news_text:
        return render_template(
            "analyzer.html",
            error="Please enter some news text before analyzing."
        )

    try:
        # Apply training-compatible preprocessing
        cleaned_text = clean_text(news_text)

        if not cleaned_text:
            return render_template(
                "analyzer.html",
                error="The entered text contains no usable content.",
                news_text=news_text
            )

        # TF-IDF feature extraction
        text_vector = vectorizer.transform([cleaned_text])

        # Generate prediction
        prediction = model.predict(text_vector)[0]

        # Calculate predicted-class probability
        probabilities = model.predict_proba(text_vector)[0]
        class_index = list(model.classes_).index(prediction)
        confidence = probabilities[class_index] * 100

        result = "FAKE NEWS" if prediction == 0 else "REAL NEWS"

        return render_template(
            "analyzer.html",
            prediction=result,
            confidence=round(float(confidence), 2),
            news_text=news_text
        )

    except Exception:
        app.logger.exception("Prediction failed")

        return render_template(
            "analyzer.html",
            error="An error occurred while analyzing the text. Please try again.",
            news_text=news_text
        )


if __name__ == "__main__":
    app.run(debug=False)
