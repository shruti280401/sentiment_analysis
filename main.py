import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# Load the model and vectorizer
with open('sentiment_analysis_model.pkl', 'rb') as file:
    clf, tfidf_vectorizer = joblib.load(file)

# Streamlit app
st.title('Sentiment Analysis App')
st.write('Enter the text you want to analyze:')

# Text input
user_input = st.text_area('Text Input', '')

# Predict button
if st.button('Predict'):
    if user_input:
        # Transform the user input using the loaded vectorizer
        user_input_tfidf = tfidf_vectorizer.transform([user_input])
        
        # Predict the sentiment
        prediction = clf.predict(user_input_tfidf)
        st.write(f'Prediction: {prediction}') 
        # Display the prediction
        if prediction[0] == 1:
            st.success('The sentiment of the text is positive.')
        else:
            st.error('The sentiment of the text is negative.')
    else:
        st.error('Please enter some text to analyze.')


