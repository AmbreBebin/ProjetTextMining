# -*- coding: utf-8 -*-

### NETTOYAGE DONNEES - DEFINITION DES FONCTIONS

import re
import nltk
import spacy
import pandas as pd
from nltk.corpus import stopwords

#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('stopwords')


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
stop_words.append("avoir")
stop_words.append("être")
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
    text = suppr_ponct(text)
    text = lemmatisation(text)
    text = stop_words_function(text)
    text = [elem.replace('ia', 'intelligence artificielle') for elem in text]
    return text

def nettoyage_phrase(text):
    text = nettoyage_regex(text)
    text = tokenisation(text)
    return text


### RECUPERATON DES DONNEES 


import snscrape.modules.twitter as sntwitter 


maxTweets = 15


tweets_list = [] 
tweets_phrase = []
# Using TwitterSearchScraper to scrape data and append tweets to list 
for i,tweet in enumerate(sntwitter.TwitterSearchScraper(query="chatgpt lang:fr since:2022-11-15 until:2022-12-15").get_items()):
    if i>maxTweets:
        break
    tweets_list.append(' '.join(nettoyage([tweet.content][0])))
    tweets_phrase.append(' '.join(nettoyage_phrase([tweet.content][0])))
    
    
tweets_df_before = pd.DataFrame(tweets_list, columns=['Tweet'])
tweets_df_before = tweets_df_before.dropna()
tweets_df_before = tweets_df_before.reset_index(drop=True)

tweets_df_before_phrase = pd.DataFrame(tweets_phrase, columns=['Tweet'])
tweets_df_before_phrase = tweets_df_before_phrase.dropna()
tweets_df_before_phrase = tweets_df_before_phrase.reset_index(drop=True)



tweets_list = []
tweets_phrase = []
# Using TwitterSearchScraper to scrape data and append tweets to list 
for i,tweet in enumerate(sntwitter.TwitterSearchScraper(query="chatgpt lang:fr since:2023-01-10 until:2023-02-10").get_items()):
    if i>maxTweets:
        break
    tweets_list.append(' '.join(nettoyage([tweet.content][0])))
    tweets_phrase.append(' '.join(nettoyage_phrase([tweet.content][0])))
    
    
tweets_df_now = pd.DataFrame(tweets_list, columns=['Tweet'])
tweets_df_now = tweets_df_now.dropna()
tweets_df_now = tweets_df_now.reset_index(drop=True)

tweets_df_now_phrase = pd.DataFrame(tweets_phrase, columns=['Tweet'])
tweets_df_now_phrase = tweets_df_before_phrase.dropna()
tweets_df_now_phrase = tweets_df_before_phrase.reset_index(drop=True)



tweets_df_now.to_csv("data_chatgpt_now_nettoye.csv", index = False)
tweets_df_before.to_csv("data_chatgpt_before_nettoye.csv", index = False)
tweets_df_now_phrase.to_csv("data_chatgpt_now_phrase.csv", index = False)
tweets_df_before_phrase.to_csv("data_chatgpt_before_phrase.csv", index = False)