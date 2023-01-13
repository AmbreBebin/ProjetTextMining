# -*- coding: utf-8 -*-

import snscrape.modules.twitter as sntwitter 
import pandas as pd 


tweets_list1 = [] 
maxTweets = 20000
# Using TwitterSearchScraper to scrape data and append tweets to list 
for i,tweet in enumerate(sntwitter.TwitterSearchScraper(query="chatgpt lang:fr").get_items()):
    if i>maxTweets:
        break
    tweets_list1.append([tweet.content])
    
    
tweets_df2 = pd.DataFrame(tweets_list1, columns=['Text'])




tweets_df2.to_csv("data_chatgpt.csv", index = False, encoding="utf-8")
