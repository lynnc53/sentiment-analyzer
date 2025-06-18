# for streamlit application 
import streamlit as st 
from scripts.inference import predict_sentiment # from scripts/inference.py import predict_sentiment function 

st.title('Movie Review Sentiment Analysis')

# save input text in the variable review_text
review_text = st.text_input("Enter your movie review", "Type your review...")

# Display the result when button is clicked 
if st.button("Predict Sentiment"):
    result = predict_sentiment(review_text)
    st.write(f"Sentiment: {result}")
