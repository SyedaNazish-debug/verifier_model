

from flask import  Flask, render_template, request
import joblib
import os

app = Flask(__name__)

BASE_DIR =os.path.dirname(os.path.abspath(__file__))
model= joblib.load(
    os.path.join("model/VERIFIER LENSE MODEL.pkl")
)

vectorizer = joblib.load(
    os.path.join(BASE_DIR,"model","TF-IDF_VECTORIZIER.pkl"))

@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict", methods=["POST"])
def predict():
    news_text = request.form.get("news_text")
    text_vector = vectorizer.transform([news_text])
    prediction = model.predict(text_vector)[0]
    confidence = max(model.predict_proba(text_vector)[0])*100

    if prediction == 1:
        result = "FAKE NEWS"
    else:
        result = "REAL NEWS"
    return render_template(
        "index.html",
        prediction=result,
        confidence=round(confidence,2),
        news_text=news_text
    )

if __name__ == "__main__":
    app.run(debug=True)