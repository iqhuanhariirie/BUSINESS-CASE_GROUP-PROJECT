from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

app = Flask(__name__)

# Load the model components
try:
    with open('model/svm_model.pkl', 'rb') as f:
        model_components = pickle.load(f)
        # Extract model components
        weights = model_components['weights']
        bias = model_components['bias']
        vectorizer = model_components['vectorizer']
    print("Model components loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")

def preprocess_text(text):
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Clean text
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenize and lemmatize
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words 
             if word not in stopwords.words('english')]
    
    return ' '.join(words)

def predict_sentiment(text_vector):
    # Use the saved weights and bias for prediction
    result = np.dot(text_vector, weights) + bias
    prediction = np.sign(result)
    
    # Calculate confidence using sigmoid function
    confidence = 1 / (1 + np.exp(-abs(float(result))))  # Convert to probability between 0 and 1
    confidence = float(confidence * 100)  # Convert to percentage
    
    return prediction, confidence

# Add the home route
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            text = request.form['text']
            cleaned_text = preprocess_text(text)
            text_vector = vectorizer.transform([cleaned_text]).toarray()
            
            # Make prediction
            prediction, confidence = predict_sentiment(text_vector)
            
            # Round confidence to 2 decimal places
            result = {
                'text': text,
                'sentiment': 'Positive' if prediction > 0 else 'Negative',
                'confidence': f'{confidence:.1f}%'
            }
            
            return render_template('result.html', result=result)
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
    
