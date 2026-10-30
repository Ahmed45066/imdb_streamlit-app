import streamlit as st
from transformers import pipeline

st.title("imdb review classifier")

@st.cache_resource
def load_model():
    return pipeline("text-classification", model="AhmedRafiq/my-imdb-model")

clf = load_model()

text = st.text_area("write a movie review")

if st.button("check"):
    result = clf(text)[0]
    label = "positive" if result["label"] == "LABEL_1" else "negative"
    st.write(label, round(result["score"], 3))
