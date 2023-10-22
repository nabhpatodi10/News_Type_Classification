import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import joblib

data=[]
with open('News_Category_Dataset_v3.json', 'r') as file:
    for line in file:
        record = json.loads(line)
        data.append(record)

dataset = pd.DataFrame(data)

dataset=dataset.drop(['link', 'authors', 'date'], axis=1)

dataset['combined_text'] = dataset['headline'] + ' ' + dataset['short_description']

tfidf_vectorizer = TfidfVectorizer(max_features=84829)

tfidf_vectorizer.fit(dataset['combined_text'])

joblib.dump(tfidf_vectorizer, "Vectorizer.pkl")