# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 10:39:29 2023

@author: ben35
"""



"""
Ici, text signifie un unique tweet

"""

text = "Quand je suis en pleine crise, Tangi m'a énervé ! #ChatGPT @TanguyTallec"

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer


nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')



### NETTOYAGE REGEX
def nettoyage_regex(text):
    text = re.sub(r'http\S+', '', text)  # Enlève les urls
    text = re.sub(r'@\S+', '', text)  # Enlève les nom d'utilisateurs
    text = re.sub(r'#\S+', '', text)  # Enlève les hashtags
    
    return text

text = nettoyage_regex(text)

### TOKENISATION
def tokenisation(text):
    tokenizer = nltk.tokenize.WordPunctTokenizer()
    tweet_tokens = [tokenizer.tokenize(text)]
    return tweet_tokens[0]


text = tokenisation(text)


### STOP WORDS
stop_words = set(stopwords.words('french'))
def stop_words_function(text):
    return [w.lower() for w in text if not w.lower() in stop_words]

text = stop_words_function(text)

## SUPPRESSION PONCTUATION
def suppr_ponct(text):
    return [token for token in text if token.isalnum()]

text = suppr_ponct(text)


### LEMMATISATION
# Initialisation du lemmatiseur
lemmatizer = WordNetLemmatizer()

# Lemmatisation des tokens
lemmatized_tokens = [lemmatizer.lemmatize(token) for token in text]

print(lemmatized_tokens)


### RACINALISATION


englishStemmer = SnowballStemmer("french")

