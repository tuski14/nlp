from collections import Counter
import numpy as np

# Example corpus
corpus = [
    "the cat sat on the mat",
    "the dog chased the cat",
    "the cat chased the mouse"
]

# Preprocess the corpus (convert to lowercase and tokenize)
tokenized_corpus = [sentence.lower().split() for sentence in corpus]

# Flatten the tokenized corpus
flatten_corpus = [word for sentence in tokenized_corpus for word in sentence]

# Calculate unigram probabilities
unigram_counts = Counter(flatten_corpus)
total_words = len(flatten_corpus)
unigram_probabilities = {word: count / total_words for word, count in unigram_counts.items()}

# Function to calculate the probability of a sequence of words using unigram model
def calculate_unigram_probability(sequence):
    probability = 1.0
    for word in sequence:
        if word in unigram_probabilities:
            probability *= unigram_probabilities[word]
        else:
            # Laplace smoothing for unknown words
            probability *= 1 / (total_words + 1)
    return probability

# Example sequence
sequence = ["the", "dog"]

# Calculate the probability of the sequence
sequence_probability = calculate_unigram_probability(sequence)
print("Probability of the sequence '{}': {:.6f}".format(" ".join(sequence), sequence_probability))
