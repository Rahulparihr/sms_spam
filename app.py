import streamlit
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import word_tokenize

model = pickle.load(open('model.pkl', 'rb'))
tf = pickle.load(open('tf_vector.pkl', 'rb'))
ps = PorterStemmer()


def transform(text):
    text = text.lower()
    text = word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()
    for msg in text:
        y.append(ps.stem(msg))
    return " ".join(y)

streamlit.title('Email/SMS spam Classifier')

input_st=streamlit.text_area('enter the message')

if streamlit.button('predict'):
    transform_t=transform(input_st)
    vector=tf.transform([transform_t])
    result=model.predict(vector)[0]

    if result==1:
        streamlit.header('spam')
    else:
        streamlit.header('not spam')