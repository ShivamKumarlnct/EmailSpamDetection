
import streamlit as st
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Download NLTK stopwords
@st.cache_resource
def load_nltk():
    nltk.download("stopwords", quiet=True)

load_nltk()

# Text preprocessing function
def text_processing(text):
    text = "".join([char for char in text if char not in string.punctuation])
    words = [
        word for word in text.split()
        if word.lower() not in stopwords.words("english")
    ]
    return words

# Train model
@st.cache_resource
def train_model():
    df = pd.read_csv("spam_or_not_spam.csv")
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    cv = CountVectorizer(analyzer=text_processing)
    X = cv.fit_transform(df["email"])
    y = df["label"]

    model = MultinomialNB()
    model.fit(X, y)

    return cv, model

try:
    cv, classifier = train_model()
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# UI
st.title("📧 Email Spam Detection App")
st.write(
    "This application classifies emails as Spam or Not Spam using Machine Learning."
)

user_input = st.text_area(
    "Paste or type your email content:",
    height=200
)

if st.button("Analyze Email"):
    if user_input.strip() == "":
        st.warning("Please enter an email.")
    else:
        data = cv.transform([user_input])
        prediction = classifier.predict(data)

        if prediction[0] == 1:
            st.error("🚨 This email is SPAM.")
        else:
            st.success("✅ This email is NOT SPAM.")
