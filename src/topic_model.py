import re
import pandas as pd
import spacy
import gensim
from gensim import corpora
import pyLDAvis
import pyLDAvis.gensim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

class newdata:

    def __init__(self, data, nlp):
        self.data    = data
        self.nlp     = spacy.load('en_core_web_lg')
    
    def process_text(self, text):

        # Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
        
        # Process text with spaCy
        doc = self.nlp(text)
        tokens = [token for token in doc if not token.is_stop and token.is_alpha]
        lemmatized_tokens = [token.lemma_.lower() for token in tokens]

        return " ".join(lemmatized_tokens)
    
    def add_proctext(self):
        preproc_texts = [self.process_text(article) for article in self.data["content_trans"].to_list()]
        self.data["cleaned_text"] = preproc_texts
        return self.data