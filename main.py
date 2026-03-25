import pandas as pd

fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

print(fake.head())
print(true.head())

fake['label'] = 0   # fake
true['label'] = 1   # real

data = pd.concat([fake, true])

print(data.head())
print(data.shape)

data = data[['text', 'label']]
print(data.head())

print(data['label'].value_counts())

data = data.sample(frac=1).reset_index(drop=True)

print(data.shape)

import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

def preprocess_text(text):
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove punctuation & numbers
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    
    # 3. Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    
    # 4. Tokenize
    words = text.split()
    
    # 5. Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # 6. Stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    
    return " ".join(words)

data['text']=data['text'].apply(preprocess_text)
print(data.head())
data.to_csv("cleaned_data.csv", index=False)