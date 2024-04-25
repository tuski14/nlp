#news data 
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from langdetect import detect
import pandas as pd

# Download NLTK resources
nltk.download('punkt')

# Load the News category dataset
df = pd.read_json('News_Category_Dataset_v2.json', lines=True)

# Sample headlines for demonstration
headlines = df['headline'].sample(5, random_state=42)

# Perform language detection, word count, sentence count, and word-level tokenization for each headline
for i, headline in enumerate(headlines):
    print(f"Headline {i + 1}: {headline}")
    
    # Language detection
    try:
        language = detect(headline)
        print(f"Language: {language}")
    except:
        print("Language detection failed.")
    
    # Word count
    words = word_tokenize(headline)
    word_count = len(words)
    print(f"Word Count: {word_count}")
    
    # Sentence count
    sentences = sent_tokenize(headline)
    sentence_count = len(sentences)
    print(f"Sentence Count: {sentence_count}")
    
    # Word-level tokenization
    print("Word-level Tokenization:")
    for j, word in enumerate(words):
        print(f"   {j + 1}. {word}")
    
    print("\n")






# alternate 

import pandas as pd
from langdetect import detect
from nltk.tokenize import sent_tokenize, word_tokenize

# Load the News category dataset
df = pd.read_json('News_Category_Dataset.json', lines=True)

# Implement language detection
def detect_language(text):
    try:
        return detect(text)
    except:
        return "Unknown"

# Perform word count
def word_count(text):
    tokens = word_tokenize(text)
    return len(tokens)

# Perform sentence count
def sentence_count(text):
    sentences = sent_tokenize(text)
    return len(sentences)

# Implement sentence-level tokenization
def tokenize_sentences(text):
    sentences = sent_tokenize(text)
    tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]
    return tokenized_sentences

# Example usage
sample_text = df.headline.iloc[0]  # Taking the headline of the first row as an example
print("Sample Text:", sample_text)
print("Language:", detect_language(sample_text))
print("Word Count:", word_count(sample_text))
print("Sentence Count:", sentence_count(sample_text))
print("Tokenized Sentences:", tokenize_sentences(sample_text))
