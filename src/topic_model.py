import re
import spacy
import gensim
from gensim import corpora
import pyLDAvis
import pyLDAvis.gensim

class newsData:

    def __init__(self, data, target_col):
        """
        Initialize the newsData class.
        Parameters
        ----------
        data : DataFrame
            The input data containing the text to be processed.
        target_col : str
            The column name in the DataFrame that contains the text.
        """
        
        self.data = data
        self.nlp  = spacy.load('en_core_web_lg')
        self.data["cleaned_text"] = [
            self._process_text(article) 
            for article in self.data[target_col].to_list()
        ]
        self.corpus, self.dictionary = self._extract_corpora()
    

    def _process_text(self, text):
        """
        Preprocess the text by removing URLs, stop words, and lemmatizing.

        Parameters
        ----------
        text : str
            The text to be processed.

        Returns
        -------
        text : str
            The processed text.
        """

        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
        
        doc = self.nlp(text)
        tokens = [token for token in doc if not token.is_stop and token.is_alpha]
        lemmatized_tokens = [token.lemma_.lower() for token in tokens]

        return " ".join(lemmatized_tokens)
    
    
    def _extract_corpora(self):
        """
        Extracts the corpus and dictionary from the cleaned text.

        Returns
        -------
        corpus : list
            List of tuples containing the word id and its frequency.
        dictionary : Dictionary
            Dictionary mapping word ids to words.
        """

        input_text = self.data.cleaned_text.to_list()
        tokens     = [text.split() if isinstance(text, str) else text for text in input_text]
        dictionary = corpora.Dictionary(tokens)
        corpus = [dictionary.doc2bow(text) for text in tokens]

        return corpus, dictionary
    

    def train_lda(self, ntopics):
        """
        Train LDA model and save the visualization to an HTML file.

        Parameters
        ----------
        ntopics : int
            Number of topics to extract.
        
        Returns
        -------
        bool
            True if the model was trained successfully. 
        """

        lda_model = gensim.models.ldamodel.LdaModel(
            self.corpus, 
            num_topics = ntopics, 
            id2word    = self.dictionary, 
            passes     = 15
        )
    
        lda = pyLDAvis.gensim.prepare(lda_model, self.corpus, self.dictionary)
        pyLDAvis.save_html(lda, "ldavis/lda.html")

        return True
