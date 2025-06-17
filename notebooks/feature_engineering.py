# Load clean data
import pandas as pd 

df = pd.read_csv("data/clean_movie_reviews.csv")
print(df.head())

# # Apply Stemming 
# from nltk.stem import PorterStemmer
# stemmer = PorterStemmer()
# def stem_text(text):
#     return " ".join([stemmer.stem(word) for word in text.split()])

# df['stemmed_text'] =df['clean_text'].apply(stem_text)

# TF-IDF Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer

# set-up TF_IDF
vectorizer = TfidfVectorizer(
    max_features = 5000,
    ngram_range = (1,2),
    min_df = 5,
    max_df = 0.8,
)

x = vectorizer.fit_transform(df['clean_text'])
y = df['label']

# Save the vectorized data 
from sklearn.preprocessing import LabelEncoder
import joblib 
import numpy as np 

# encode labels 
le = LabelEncoder()
# converts text labels like positive and negative to numerical values
y_encoded = le.fit_transform(y)
# necessary cuz most ml models dont work with strings 

# Save TF_IDF vectorizer 
joblib.dump(vectorizer,"data/vectorizer.pkl")

# save X and Y 
np.save("data/X.npy", x.toarray())
np.save("data/y.npy", y_encoded)

print("Vectorizer and feature arrays saved successfully.")