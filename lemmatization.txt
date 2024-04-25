import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

# Sample text for demonstration
text = "The quick brown foxes are jumping over the lazy dogs"

# Tokenize the text into words
words = word_tokenize(text)

# Initialize the WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

# Lemmatize each word
lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

# Display the original words and their lemmatized forms
for original_word, lemmatized_word in zip(words, lemmatized_words):
    print(f"{original_word} -> {lemmatized_word}")
