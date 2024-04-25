# using nltk 

import nltk
from nltk.tokenize import word_tokenize
from collections import Counter

# Download NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Sample text for demonstration
text = "The quick brown foxes are jumping over the lazy dogs"

# Tokenize the text into words
words = word_tokenize(text)

# Perform POS tagging
pos_tags = nltk.pos_tag(words)

# Display POS tags
print("POS Tags:")
for word, pos_tag in pos_tags:
    print(f"{word}: {pos_tag}")

# Calculate frequency of POS tags
pos_tag_freq = Counter(tag for word, tag in pos_tags)

# Display frequency list of POS tags
print("\nFrequency of POS Tags:")
for tag, count in pos_tag_freq.items():
    print(f"{tag}: {count}")
    
    
    
#using spacy 


import spacy
from collections import Counter

# Load the English language model
nlp = spacy.load("en_core_web_sm")

# Sample text for demonstration
text = "The quick brown foxes are jumping over the lazy dogs"

# Process the text using SpaCy
doc = nlp(text)

# Perform POS tagging
pos_tags = [(token.text, token.pos_) for token in doc]

# Display POS tags
print("POS Tags:")
for word, pos_tag in pos_tags:
    print(f"{word}: {pos_tag}")

# Calculate frequency of POS tags
pos_tag_freq = Counter(tag for _, tag in pos_tags)

# Display frequency list of POS tags
print("\nFrequency of POS Tags:")
for tag, count in pos_tag_freq.items():
    print(f"{tag}: {count}")
