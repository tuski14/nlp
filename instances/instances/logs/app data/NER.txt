#ner code 
import spacy
from spacy import displacy

# Load the English language model
nlp = spacy.load("en_core_web_sm")

# Example text
text = """
Apple Inc. is an American multinational technology company headquartered in Cupertino, California.
It was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976.
"""

# Perform Named Entity Recognition (NER)
doc = nlp(text)

# Display the named entities
print("Named Entities:")
for ent in doc.ents:
    print(ent.text, "-", ent.label_)

# Display named entity visualization
displacy.render(doc, style="ent", jupyter=True)