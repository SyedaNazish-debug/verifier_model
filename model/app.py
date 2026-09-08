# import streamlit as st 
# from verifer_model import detect_news,accuracy

# st.set_page_config(
#     page_title="FAKE NEWS DETECTOR",
#     page_icon="💯/❌",
#     layout="centered")

# st.title("FAKE NEWS DETECTION SYSTEM")
# st.write("Enter a news article below & the system will analyze it as Fake or Real")

# st.metric(label="Model accuracy",
#           value= f"{accuracy*100:.2f}%")

# news= st.text_area(
#     "Enter News Article",
#     height=250,
#     placeholder="paste news article here or the title"
# )

# if st.button("DETECT NEWS",use_container_width=True):
#     if not news.strip():
#         st.warning("please enter a news article.")
#     else:
#         prediction,confidence= detect_news(news)
#         if prediction == "Fake":
#             st.error("Fake News, please fact check it")
#         else:
#             st.success("Source varified, Real News confirm")

#         st.write(
#             f"**Confidence:**{confidence:.2f}%"
#         )

from flask import  Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("model/VERIFIER LENSE MODEL.pkl")
vectorizer = joblib.load("model/TF-IDF_VECTORIZER.pkl")

@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict", methods=["POST"])
def predict():
    news_text = request.form.get("news_text")
    text_vector = vectorizer.transform([news_text])
    predicton = model.predict(text_vector)[0]
    confidence = max(model.predict_proba(text_vector)[0])*100

    if predicton == 1:
        result = "FAKE NEWS"
    else:
        result = "REAL NEWS"
    return render_template(
        "index.html",
        predicton=result,
        confidence=round(confidence,2),
        news_text=news_text
    )

if __name__ == "__main__":
    app.run(debug=True)