from flask import Flask, request, render_template
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('stopwords', quiet=True)

# load saved model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

app = Flask(__name__)

# preprocessing function (same as before)
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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        text = request.form.get('news', '')

        # handle empty input
        if not text:
            return render_template('index.html', prediction_text="Please enter some text")

        # minimum length check
        if len(text.split()) < 20:
            return render_template('index.html', prediction_text="Please enter a full news article (at least 20 words)")

        processed = preprocess_text(text)

        vectorized = vectorizer.transform([processed])

        prediction = model.predict(vectorized)[0]
        proba = model.predict_proba(vectorized)[0]
        confidence = max(proba)

        label = "Real News" if prediction == 1 else "Fake News"
        result = f"{label} ({confidence*100:.2f}% confidence)"

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)