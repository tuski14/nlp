import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())

    # Remove punctuation and stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]

    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return tokens

def identify_sentiment(text):
    sid = SentimentIntensityAnalyzer()
    scores = sid.polarity_scores(text)

    if scores['compound'] >= 0.05:
        return 'Positive'
    elif scores['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'
    
def identify_emotions(text):
    emotions = []

    # Define emotion keywords or
    emotion_keywords = {
        'anger': ['angry', 'frustrated', 'irritated'],
        'joy': ['happy', 'joyful', 'ecstatic'],
        'sadness': ['sad', 'depressed', 'gloomy'],
        'fear': ['fear', 'anxious', 'scared'],
        'surprise': ['surprise', 'shock', 'astonish']
    }

    # Preprocess text
    tokens = preprocess_text(text)

    # Identify emotions
    for token in tokens:
        for emotion, keywords in emotion_keywords.items():
            if token in keywords:
                emotions.append(emotion)

    return list(set(emotions))
def main():
    # Input text
    text = input("Enter the text: ")

    # Identify sentiment
    sentiment = identify_sentiment(text)
    print(text)
    print("Sentiment:", sentiment)

    # Identify emotions
    emotions = identify_emotions(text)
    print("Emotions:", emotions)

if __name__ == "__main__":
    main()
