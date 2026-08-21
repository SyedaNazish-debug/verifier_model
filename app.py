import streamlit as st 
from verifer_model import detect_news,accuracy

st.set_page_config(
    page_title="FAKE NEWS DETECTOR",
    page_icon="💯/❌",
    layout="centered")

st.title("FAKE NEWS DETECTION SYSTEM")
st.write("Enter a news article below & the system will analyze it as Fake or Real")

st.metric(label="Model accuracy",
          value= f"{accuracy*100:.2f}%")

news= st.text_area(
    "Enter News Article",
    height=250,
    placeholder="paste news article here or the title"
)

if st.button("DETECT NEWS",use_container_width=True):
    if not news.strip():
        st.warning("please enter a news article.")
    else:
        prediction,confidence= detect_news(news)
        if prediction == "Fake":
            st.error("Fake News, please fact check it")
        else:
            st.success("Source varified, Real News confirm")

        st.write(
            f"**Confidence:**{confidence:.2f}%"
        )