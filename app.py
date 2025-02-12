import streamlit as st
import pickle
import pandas as pd

# Load model and vectorizer
with open("models/svm_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("models/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Streamlit UI
st.title("Fake News Detection")
st.write("Enter a news article to classify it as Fake or Real.")

# User input
news_text = st.text_area("Paste news article here:")

if st.button("Predict"):
    if news_text.strip():
        # Transform text
        X_input = vectorizer.transform([news_text])
        prediction = model.predict(X_input)[0]

        # Display result
        if prediction == 1:
            st.success("✅ This news is REAL!")
        else:
            st.error("🚨 This news is FAKE!")
    else:
        st.warning("Please enter some text to classify.")
