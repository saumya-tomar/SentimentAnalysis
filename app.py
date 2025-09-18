import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Load Pretrained Model and Vectorizer
model = pickle.load(open(r'C:\Users\RentoBees\PycharmProjects\Sentiment-Analysis\trained_model.sav', 'rb'))


@st.cache_data
def load_data():
    column_names = ['target', 'id', 'date', 'flag', 'user', 'text']
    twitter_data = pd.read_csv(
        r'C:\Users\RentoBees\PycharmProjects\Sentiment-Analysis\data.csv',
        names=column_names, encoding='ISO-8859-1')
    twitter_data.replace({'target': {4: 1}}, inplace=True)
    return twitter_data


@st.cache_data
def preprocess_data(df):
    stop_words = set(stopwords.words('english'))
    port_stem = PorterStemmer()

    def stemming(content):
        stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
        stemmed_content = stemmed_content.lower()
        stemmed_content = stemmed_content.split()
        stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stop_words]
        return ' '.join(stemmed_content)

    df['stemmed_content'] = df['text'].apply(stemming)
    return df


@st.cache_resource
def get_vectorizer_and_transform(data):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data)
    return vectorizer, X


# Load and preprocess data
twitter_data = load_data()
twitter_data = preprocess_data(twitter_data)

# Train-test split (fixed seed for reproducibility)
from sklearn.model_selection import train_test_split

X = twitter_data['stemmed_content']
Y = twitter_data['target']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Vectorize test data with fitted vectorizer
vectorizer, X_train_vec = get_vectorizer_and_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Streamlit UI

st.title("Twitter Sentiment Analysis")

max_index = X_test_vec.shape[0] - 1
index = st.number_input(f"Enter tweet index (0 to {max_index}):", min_value=0, max_value=max_index, step=1)

if st.button("Predict Sentiment"):
    X_new = X_test_vec.getrow(index)
    actual_label = Y_test.iloc[index]
    prediction = model.predict(X_new)

    st.write(f"Actual Label: {'Positive' if actual_label == 1 else 'Negative'}")
    st.write(f"Predicted Sentiment: {'Positive' if prediction[0] == 1 else 'Negative'}")