import pandas as pd
import nltk
import re

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')


# 1. Load Dataset

fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Add labels
fake['label'] = 0
true['label'] = 1

data = pd.concat([fake, true])

data = data[['text', 'label']]

data = data.sample(frac=1).reset_index(drop=True)

# 2. Text Preprocessing

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    words = text.split()

    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]

    return " ".join(words)

# Apply preprocessing
data['text'] = data['text'].apply(preprocess_text)


# 3. Feature Extraction (TF-IDF)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['text'])
y = data['label']


# 4. Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Logistic Regression Model

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

print("Logistic Regression Results")
print("Accuracy:", accuracy_score(y_test, lr_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, lr_pred))
print("Classification Report:\n", classification_report(y_test, lr_pred))


# 6. Naive Bayes Model

nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

nb_pred = nb_model.predict(X_test)

print("\nNaive Bayes Results")
print("Accuracy:", accuracy_score(y_test, nb_pred))

print("\nModel Comparison Completed")