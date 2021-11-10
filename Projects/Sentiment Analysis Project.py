#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import re

# Modules for visualization
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Tools for preprocessing input data
from bs4 import BeautifulSoup
from nltk import word_tokenize
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import CountVectorizer 
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import gensim


# In[5]:


data = pd.read_csv("E:\\ML\\mlp-course-material\\MODULE 9 - Mini Projects\\2. Sentiment Analysis Kaggle\\sentiment_data.tsv", delimiter = "\t")


# In[6]:


data.head()


# In[7]:


data.info()


# In[8]:


data = data[:2000]


# In[9]:


data.head(10)


# In[10]:


data.info()


# In[20]:


data["id"].nunique(0)


# In[21]:


data.describe()


# In[22]:


data.drop(["id"], axis = 1)


# In[23]:


data.shape


# In[24]:


#Processing Messages


# In[25]:


def processing(review):

    # Remove email addresses with 'emailaddr'    
    raw_review = re.sub('\b[\w\-.]+?@\w+?\.\w{2,4}\b', " ", review)
    
    # Remove URLs with 'httpaddr'
    raw_review = re.sub('(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)', " ", raw_review) 

    # Remove non-letters        
    raw_review = re.sub("[^a-zA-Z]", " ", raw_review) 
    
    # Remove numbers
    raw_review = re.sub('\d+(\.\d+)?', " ", raw_review)

    # Convert to lower case, split into individual words
    words = raw_review.lower().split()                                             

    # Gather the list of stopwords in English Language
    stops = set(stopwords.words("english"))                  

    # Remove stop words and stemming the remaining words
    meaningful_words = [ps.stem(w) for w in words if not w in stops]   

    # Join the tokens back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))


# In[26]:


#Corpus
clean_review_corpus = []
#Initialising Poster Stemmer
ps = PorterStemmer()


# In[27]:


#Counting number of views
review_count = data["review"].size 


# In[28]:


review_count


# In[31]:


for i in range(0,review_count):
    clean_review_corpus.append(processing(data["review"][i]))


# In[32]:


data["review"][0] 


# In[33]:


clean_review_corpus[0]


# In[34]:


#Preparing vectors for each message


# In[36]:


cv = CountVectorizer()
data_input = cv.fit_transform(clean_review_corpus)
data_input = data_input.toarray()


# In[37]:


data_input[0]


# In[41]:


from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(background_color='black', stopwords=stopwords,
        max_words=200,
        max_font_size=40, 
        scale=3,
        random_state=1 # chosen at random by flipping a coin; it was heads
).generate(str(data))

    fig = plt.figure(1, figsize=(15, 15))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()

show_wordcloud(clean_review_corpus)


# In[39]:





# In[42]:


#Applying classification models


# In[46]:


data_output = data["sentiment"]
data_output.value_counts().plot.bar()


# In[47]:


#Splitting Data


# In[50]:


from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(data_input, data_output, test_size = 0.20, random_state = 0)


# In[51]:


model_nvb = GaussianNB()
model_nvb.fit(xtrain,ytrain)

model_rf = RandomForestClassifier(n_estimators=1000, random_state=0)
model_rf.fit(xtrain, ytrain)

model_dt = tree.DecisionTreeClassifier()
model_dt.fit(xtrain, ytrain)


# In[52]:


#Prediction


# In[53]:


prediction_nvb = model_nvb.predict(xtest)
prediction_rf = model_rf.predict(xtest)
prediction_dt = model_dt.predict(xtest)


# In[55]:


print ("Accuracy for Naive Bayes : %0.5f \n\n" % accuracy_score(ytest, prediction_nvb))
print ("Classification Report Naive bayes: \n", classification_report(ytest, prediction_nvb))


# In[56]:


print ("Accuracy for Decision Tree: %0.5f \n\n" % accuracy_score(ytest, prediction_dt))
print ("Classification Report Decision Tree: \n", classification_report(ytest, prediction_dt))


# In[57]:


print ("Accuracy for Random Forest: %0.5f \n\n" % accuracy_score(ytest, prediction_rf))
print ("Classification Report Random Forest: \n", classification_report(ytest, prediction_rf))


# In[ ]:




