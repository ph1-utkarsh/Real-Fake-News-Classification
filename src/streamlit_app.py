# src/streamlit_app.py
import streamlit as st
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLTK resources are available (download if needed)
try:
    stopwords.words('english')  # Try accessing stopwords to trigger download if not present
    WordNetLemmatizer()  # Try creating WordNetLemmatizer to trigger download if not present
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')

# Load the model and vectorizer with caching for faster reloads
@st.cache_resource  # Use st.cache_resource for Streamlit >= 1.12.0
def load_model():
    with open("models//model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models//vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model()

# Define a more complete preprocessing function for input text
def preprocess_input(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)


# Streamlit app layout
st.title("Fake News Detection")
st.write("Enter a news article text below to determine if it is Fake or Real:")

user_input = st.text_area("News Article Text", "")

if st.button("Predict"):
    if user_input.strip():
        processed_text = preprocess_input(user_input)
        text_tfidf = vectorizer.transform([processed_text])
        prediction = model.predict(text_tfidf)[0]
        result = "Real News" if prediction == 1 else "Fake News"
        st.write("Prediction:", result)
    else:
        st.write("Please enter some text to get a prediction.")