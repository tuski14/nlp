#text calssification movie review 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
import re

# Step 1: Data Loading and Inspection
df = pd.read_csv(r"C:\Users\shubh\Downloads\archive (5)\IMDB Dataset.csv")

print(df.head())

# Preprocess the text data
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    # Perform stemming
    ps = PorterStemmer()
    stemmed_tokens = [ps.stem(word) for word in filtered_tokens]
    # Join stemmed tokens back into text
    text = ' '.join(stemmed_tokens)
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['review'] = df['review'].apply(preprocess_text)

# Step 3: Split the Dataset
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

# Step 4: Feature Extraction
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 5: Model Training
classifier = MultinomialNB()
classifier.fit(X_train_tfidf, y_train)

# Step 6: Model Evaluation
y_pred = classifier.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# Example sentence
input_sentence = "This movie is not well directed!"

# Step 1: Preprocess the input sentence
preprocessed_sentence = preprocess_text(input_sentence)

# Step 2: Transform the preprocessed sentence using TF-IDF vectorizer
input_sentence_tfidf = vectorizer.transform([preprocessed_sentence])

# Step 3: Use the trained classifier to predict the label
predicted_label = classifier.predict(input_sentence_tfidf)

print("Predicted Label:", predicted_label)
