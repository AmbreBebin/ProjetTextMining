# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 10:39:29 2023

@author: ben35
"""



"""
Ici, text signifie un unique tweet

"""

text = "Chat (ou chatbot) est un logiciel de conversation conçu pour communiquer avec les utilisateurs en utilisant des algorithmes de reconnaissance de la parole et du langage naturel. Les chats sont souvent utilisés pour fournir une assistance en ligne aux utilisateurs, en répondant à leurs questions et en les aidant à résoudre des problèmes. Les chats peuvent être déployés sur de nombreuses plateformes différentes, telles que les sites Web, les applications mobiles, les messageries instantanées et les systèmes de centre d'appel. L'objectif principal des chats est de fournir une assistance en temps réel aux utilisateurs, en utilisant des algorithmes pour comprendre et interagir avec eux. Les chats peuvent être programmés pour être très spécifiques et pour traiter des tâches spécifiques, telles que la réservation de billets d'avion ou la vérification du statut d'une commande en ligne. Les chats peuvent également être conçus pour être plus généraux et pour aider les utilisateurs avec une variété de questions et de tâches. Les chats sont souvent conçus pour être interactifs et conviviaux, en utilisant une combinaison de textes et de médias tels que les images et les vidéos pour aider à fournir des informations aux utilisateurs. Les chats peuvent également utiliser des techniques d'apprentissage automatique pour améliorer leur compréhension des utilisateurs et de leurs besoins, ainsi que pour fournir des réponses plus précises et plus utiles. Les chats sont de plus en plus populaires dans de nombreux secteurs différents, car ils offrent une solution abordable et flexible pour fournir une assistance en ligne aux utilisateurs. Les chats peuvent être déployés rapidement et facilement, ce qui signifie qu'ils peuvent être utilisés pour résoudre rapidement des problèmes, sans nécessiter de personnel supplémentaire.En plus de fournir une assistance en ligne, les chats peuvent également être utilisés pour améliorer les opérations de service à la clientèle et les processus de vente. Les chats peuvent être programmés pour fournir des informations sur les produits et les services, pour aider les utilisateurs à trouver ce dont ils ont besoin, et pour les guider à travers le processus d'achat.En conclusion, les chats sont de plus en plus populaires pour fournir une assistance en ligne rapide et efficace aux utilisateurs. Les chats peuvent être conçus pour être très spécifiques ou plus généraux, en fonction des besoins de l'entreprise, et peuvent utiliser des algorithmes d'apprentissage automatique pour améliorer leur compré"

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
import numpy as np
import plotly.express as px 
import spacy
from spacy import displacy
from nltk.probability import FreqDist
import plotly.express as px
from wordcloud import ImageColorGenerator
from PIL import Image
from nltk.text import Text
import pandas as pd


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


### NUAGE DE MOTS 
from wordcloud import WordCloud
import matplotlib.pyplot as plt


# Define a list of tokens
tokens = lemmatized_tokens

# Create a wordcloud object
wordcloud = WordCloud().generate_from_frequencies(dict(zip(tokens, [1]*len(tokens))))

# Plot the wordcloud
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# Create a wordcloud object
wordcloud = WordCloud().generate_from_frequencies(dict(zip(tokens, [1]*len(tokens))))

# Create a list of tuples, where each tuple contains the word and its count
words = [(word, freq) for word, freq in zip(tokens, [1]*len(tokens))]

"""
# Create a scatter plot of the words
fig = px.scatter(words, x=[1]*len(tokens), y=[1]*len(tokens), text=[word for word, freq in words],
                size=[freq for word, freq in words], hoverinfo='text',
                title='Nuage de mots interactif')
fig.update_layout(xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
                  yaxis=dict(showgrid=False, showticklabels=False, zeroline=False))
fig.show()
"""


# Création générateur de couleurs (bleus foncés)
def color_func(word, font_size, position, orientation, random_state=None,
                    **kwargs):
    return "hsl(240, 100%%, %d%%)" % np.random.randint(50, 70)

# Test wordcloud 
mask = np.array(Image.open("C:/Users/Ambre/Downloads/ProjetTextMining-dev/ProjetTextMining-dev/twi1.png"))
wordcloud = WordCloud(mask=mask, background_color="white").generate_from_frequencies(dict(zip(tokens, [1]*len(tokens))))
image_colors = ImageColorGenerator(mask)
plt.imshow(wordcloud.recolor(color_func=color_func), interpolation="bilinear")
plt.axis("off")
plt.show()

# bon wordcloud : 
mask = np.array(Image.open("C:/Users/Ambre/Downloads/ProjetTextMining-dev/ProjetTextMining-dev/twi5.png"))
wordcloud = WordCloud(mask=mask, background_color="white").generate_from_frequencies(dict(zip(tokens, [1]*len(tokens))))
image_colors = ImageColorGenerator(mask)
plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear")
plt.axis("off")
plt.show()


# Graphique des dépendances  

nlp = spacy.load("fr_core_news_sm")
doc = "Le chat mange une souris."
doc = nlp(doc)
print(displacy.render(doc, style='dep', jupyter=False, options={'distance': 130})) # a faire sur jupyter : fonctionne mais verifier pertinence 

# Analyse des fréquences 

# Comptage fréquence des mots
word_counts = nltk.FreqDist(text)

# Afficher les 10 mots les plus fréquents
print(word_counts.most_common(10))
top_10 = word_counts.most_common(10)

# Graphique des fréquences 
plt.barh(list(zip(*top_10))[0], list(zip(*top_10))[1], color = "darkorange")
plt.title("Fréquence d'apparition des mots")
plt.xlabel("Fréquence")
plt.ylabel("Mots")
plt.xticks(rotation=45)
plt.gca().invert_yaxis()
plt.show()

# Visualiser les résultats sous forme de graphique
top_30 = word_counts.most_common(30)

# Word_counts.plot(30, cumulative=False, title = "Fréquence d'apparition des mots")
plt.plot(list(zip(*top_30))[0], list(zip(*top_30))[1], color = "darkorange")
plt.title("Fréquence d'apparition des mots")
plt.xlabel("Mots")
plt.ylabel("Fréquence")
plt.xticks(rotation=90)
plt.show()

# Test avec plotly => Voir résltats sur Jupyter 
fig = px.bar(list(zip(*top_10))[0], list(zip(*top_10))[1])
fig.show()

# Contexte d'apparition des mots 
contexte = Text(text)
contexte.concordance("être") # renvoie resultats similaires à ceux du poly 

concordance = []
for word in contexte.vocab().keys():
    for val in contexte.concordance_list(word):
        concordance.append((word, val[0], val[1]))

df = pd.DataFrame(concordance, columns=["Mot", "Context", "Index"])
df.head() # bof resultat pas super beau 

