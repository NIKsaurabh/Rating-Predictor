import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer ,TfidfTransformer
import pickle
import streamlit as st
nltk.download('punkt')
nltk.download('stopwords')

pkl = open('model.pkl','rb')
model = pickle.load(pkl)

def predict_rating(text):
    
    stop_words = set(stopwords.words('english'))
    stop_words.remove('not')
    stop_words.remove('nor')

    ps = PorterStemmer()
    text = re.sub('[^A-Za-z]',' ',text)
    text = text.lower()
    word = nltk.word_tokenize(text)
    word = [ps.stem(j) for j in word if j not in stop_words]
    clean_text = ' '.join(word)
    model = pickle.load(open('model.pkl','rb'))
    tfidf = TfidfVectorizer(vocabulary=pickle.load(open("tfidf.pkl", "rb")))
    transformer = TfidfTransformer()
    temp = transformer.fit_transform(tfidf.fit_transform([clean_text]))
    prediction = model.predict(temp)[0]
    return prediction

def main():
    st.title('Rating Predictor')
    review = st.text_input('Enter Review')
    rating = st.number_input('Enter Rating',1,5, value=1)
    result = ''
    if st.button('Submit'):
        result = predict_rating(review)
        if rating == 3:
            st.success("Your review is submitted")
        elif result == 1 and rating < 3:
            st.text('Your review is positive and rating is less, please reconsider your rating!')
        elif result == 0 and rating >3:
            st.text('Your review is negative and rating is more, please reconsider your rating!')
        else:
            st.success("Your review is submitted")
    #st.success('Output is {}'.format(result))

if __name__ ==  '__main__':
    main()

