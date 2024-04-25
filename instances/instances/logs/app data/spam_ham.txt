import pandas as pd
import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the SMS Spam Collection dataset
df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]  # Keep only the label and text columns
df.columns = ['label', 'text']  # Rename columns for clarity

# Preprocess the text data
def preprocess_text(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in tokens]  # Apply stemming
    cleaned_text = ' '.join(stemmed_tokens)
    return cleaned_text

# Apply preprocessing to the text column
df['text'] = df['text'].apply(preprocess_text)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000)  # Limiting to top 5000 features
X = vectorizer.fit_transform(df['text'])
y = df['label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Multinomial Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Function to classify input text
def classify_text(text):
    cleaned_text = preprocess_text(text)
    text_vectorized = vectorizer.transform([cleaned_text])
    prediction = classifier.predict(text_vectorized)[0]
    return prediction

# Example input text
input_text = "Congratulations! You've won a free vacation. Click here to claim your prize."

# Classify the input text
classification = classify_text(input_text)
print("\nClassification for input text:")
print("Predicted label:", classification)
