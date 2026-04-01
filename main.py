import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['text'])
y = data['label']
print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(X_train.shape)
print(X_test.shape)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(confusion_matrix(y_test, y_pred))