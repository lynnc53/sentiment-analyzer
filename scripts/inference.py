import joblib
import numpy as np
import pandas as pd
from nltk.corpus import stopwords 
from nltk.tokenize import RegexpTokenizer

# load model and vectorizer 
model = joblib.load("models/logistic_model.pkl")
vectorizer = joblib.load("data/vectorizer.pkl")

# define preprocessing steps 
stop_words = set(stopwords.words('english'))
tokenizer = RegexpTokenizer(r"\w+")

def preprocess(text):
    tokens = tokenizer.tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    return " ".join(tokens)

# define predict_sentiment() function 
def predict_sentiment(review_text):
    cleaned = preprocess(review_text)
    vec = vectorizer.transform([cleaned])
    # model.predict() returns a list of predictions, we take the first one
    pred = model.predict(vec)[0]
    label = "Positive" if pred == 1 else "Negative"
    return label 

# Test the function 
if __name__ == "__main__":
    sample_reviews = [      
        "An absolute masterpiece! I loved every moment of it.",
        "I did not enjoy this movie at all. It was too long and the pacing was terrible.",
        "I think this movie is fantastic! The acting was superb and the plot was engaging.",
        "This film was a complete waste of time. The story was boring and the characters were flat."
    ]

    results = []

    for review in sample_reviews:
        prediction = predict_sentiment(review)
        results.append({"review":review,"prediction":prediction})
        print(f"\nReview: {review}\n Prediction: {prediction}")

    df = pd.DataFrame(results)
    df.to_csv("data/inference_results.csv", index = False)

# Test Results Summary:
# - Tested 4 custom reviews
# - 3/4 predictions correct
# - Approx. 75% accuracy