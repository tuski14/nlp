import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download NLTK resources
nltk.download('punkt')

# Sample text for demonstration
text = "The quick brown foxes are jumping over the lazy dogs"

# Tokenize the text into words
words = word_tokenize(text)

# Initialize the Porter stemmer
stemmer = PorterStemmer()

# Perform stemming on each word
stemmed_words = [stemmer.stem(word) for word in words]

# Display the original words and their stemmed forms
for original_word, stemmed_word in zip(words, stemmed_words):
    print(f"{original_word} -> {stemmed_word}")