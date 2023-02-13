# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 10:39:29 2023

@author: ben35
"""



"""
Ici, text signifie un unique tweet

"""

import re
import nltk
import spacy
import pandas as pd
from nltk.corpus import stopwords



#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('stopwords')


text = "Quand je suis en pleine crise, Tangi m'a énervé ! #ChatGPT @TanguyTallec"

### NETTOYAGE REGEX
def nettoyage_regex(text):
    text = re.sub(r'http\S+', '', text)  # Enlève les urls
    text = re.sub(r'@\S+', '', text)  # Enlève les nom d'utilisateurs
    text = re.sub(r'#\S+', '', text)  # Enlève les hashtags
    
    return text

### TOKENISATION
def tokenisation(text):
    tokenizer = nltk.tokenize.WordPunctTokenizer()
    tweet_tokens = [tokenizer.tokenize(text)]
    return tweet_tokens[0]

### STOP WORDS
stop_words = stopwords.words('french')
stop_words.append("chatgpt")
stop_words.append("chat")
stop_words.append("gpt")
stop_words = set(stop_words)
def stop_words_function(text):
    return [w.lower() for w in text if not w.lower() in stop_words]

## SUPPRESSION PONCTUATION
def suppr_ponct(text):
    return [token for token in text if token.isalnum()]

### LEMMATISATION
nlp = spacy.load('fr_core_news_md')
def lemmatisation(text):
    lem = []
    for i in range(len(text)):
        doc = nlp(text[i])
        for token in doc:
            lem.append(token.lemma_)
    return lem


def nettoyage(text):
    text = nettoyage_regex(text)
    text = tokenisation(text)
    text = stop_words_function(text)
    text = suppr_ponct(text)
    text = lemmatisation(text)
    
    return text

text = "Aujourd'hui, mon lapin est sortie de sa cage. Il a voulu s'échapper pour aller manger des carottes! ;)"

text = nettoyage_regex(text)
text = tokenisation(text)
text = stop_words_function(text)
text = suppr_ponct(text)
text = lemmatisation(text)



# Importation données
data_now = pd.read_csv("data_chatgpt.csv", sep=';', encoding = "utf-8")
data_before = pd.read_csv("data_chatgpt_before.csv", sep=';', encoding = "utf-8")

# Boucle de nettoyage
list_tweet_now = []
for i in range(len(data_now)) :
    list_tweet_now.append(nettoyage(data_now.iloc[i, 0]))

list_tweet_before = []
for i in range(len(data_before)) :
    list_tweet_before.append(nettoyage(data_before.iloc[i, 0]))
