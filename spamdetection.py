import streamlit as st
import numpy as np
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# 1. Download required natural language assets
@st.cache_resource
def load_nltk():
    nltk.download('stopwords')

load_nltk()

# 2. Text Processing Function matching your model
def text_processing(text):
    rm_punc = [char for char in text if char not in string.punctuation]
    rm_punc = ''.join(rm_punc)
    clean_words = [word for word in rm_punc.split() if word.lower() not in stopwords.words('english')]
    return clean_words

# 3. Load Dataset and Train the Model
@st.cache_resource
def train_model():
    # Reads the dataset uploaded in your repository
    data_frame = pd.read_csv('spam_or_not_spam.csv')
    data_frame.drop_duplicates(inplace=True)
    data_frame.dropna(inplace=True)
    
    # Vectorize text mapping your custom layout
    cv = CountVectorizer(analyzer=text_processing)
    X = cv.fit_transform(data_frame['email'])
    y = data_frame['label']
    
    # Train Naive Bayes Classifier
    classifier = MultinomialNB()
    classifier.fit(X, y)
    return cv, classifier

# Initialize components
try:
    cv, classifier = train_model()
except Exception as e:
    st.error(f"Error loading dataset file: {e}. Please ensure 'spam_or_not_spam.csv' is present in your repository root.")

# 4. Streamlit Web User Interface Design
st.title("📧 Email Spam Detection App")
st.write("This web application uses a Multinomial Naive Bayes model to classify messages as Spam or Not Spam.")

# User text input block
user_input = st.text_area("Paste or type the email content below:", height=150)

if st.button("Analyze Email"):
    if user_input.strip() == "":
        st.warning("Please enter some text to check.")
    else:
        # Convert input text into token format matching your training feature space
        data = cv.transform([user_input])
        prediction = classifier.predict(data)
        
        # Display classification result
        if prediction[0] == 1:
            st.error("🚨 Warning: This email is classified as SPAM!")
        else:
            st.success("✅ Safe: This email is classified as NOT SPAM.")
