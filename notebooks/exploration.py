import nltk # natural language toolkit
import pandas as pd # tables and data manipulation
import random # to shuffle data 
import matplotlib.pyplot as plt # plotting
import seaborn as sns # plotting
from nltk.corpus import stopwords # list of enlgish filler words
from nltk.corpus import movie_reviews # 2000 movie reviews labeled as positive or negative
from nltk.tokenize import RegexpTokenizer # breaks text into word-level tokens
import string 

tokenzier = RegexpTokenizer(r'\w+') # tokenizer that removes punctuation

# fileids = movie_reviews.fileids()
# print(fileids[:10])
# raw = movie_reviews.raw(fileids[0])
# print(raw[:1000])  

## load data 
# loops through all file IDs in each category (positive and negative)
# and returns a list of tuples where each tuple contains the raw text of the review and its category
documents = [(movie_reviews.raw(fileid),category)
            for category in movie_reviews.categories()
            for fileid in movie_reviews.fileids(category)]

random.shuffle(documents) # shuffle the documents to ensure randomness

# converts the list of tuples into a pandas DataFrame
df = pd.DataFrame(documents, columns=['text','label']) 
# print(df.head())

# map labels to more descriptive names
df['label'] = df['label'].map({'pos':"positive", 'neg':"negative"}) 

## Explore the Data
# checking distribution of labels
sns.countplot(data=df, x='label')
plt.title('Label Distribution')
plt.show()

# summarize the distribution of review lengths
df['text_length'] = df['text'].apply(lambda x: len(x.split())) # add a column for text length
print("Review length stats:")
print(df['text_length'].describe())

# prints the first 300 characters of a sample from each label
for label in df['label'].unique():
    print(f"\n---{label.upper()} SAMPLE---")
    print(df[df["label"] == label]["text"].iloc[0][:300])

## Clean the Text
stop_words = set(stopwords.words('english')) # get the list of stop words
# print(stop_words)

def clean_text(text):
    tokens = tokenzier.tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    return " ".join(tokens)

# This function converts text to lowercase, tokenizes it, removes punctuation 
# and stop words and joins the tokens back into a single string.

df['clean_text'] = df['text'].apply(clean_text)

## Save Cleaned Dataset 
df[['clean_text','label']].to_csv("data/clean_movie_reviews.csv", index = False )
print("Cleaned dataset saved to 'data/clean_movie_reviews.csv'")