#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd


# In[1]:


f = open('mediaeval-2015-trainingset.txt').read().split('\n')


# In[2]:


f[1].split('\t')
headers = f[0].split('\t')
data_list = []
for tweet in f:
    t = tweet.split('\t')
    data_list.append(t)
import pandas as pd
df = pd.DataFrame(columns=headers)
for i in range(len(data_list)):
   df.loc[i] = data_list[i]
df = df.iloc[1:]


# In[3]:


df.head()


# In[5]:


df.shape


# In[6]:


fake = 0
true = 0
for index, row in df.iterrows():
   if row['label'] == 'fake':
       fake = fake + 1
   else:
      true = true + 1


# In[7]:


fake, true


# In[8]:


fake/true


# In[9]:


2564/1217


# In[2]:


f1 = open('mediaeval-2015-trainingset.txt').read().split('\n')
f2 = open('mediaeval-2015-testset.txt').read().split('\n')


# In[3]:


headers = f1[0].split('\t')


# In[4]:


f = f1[1:] + f2[1:]


# In[5]:


data_list = []

for tweet in f:
    t = tweet.split('\t')
    data_list.append(t)

df = pd.DataFrame(columns=headers)

for i in range(len(data_list)):
   df.loc[i] = data_list[i]

df = df.iloc[1:]


# In[6]:


df.head()


# In[7]:


df.shape


# In[8]:


fake = 0
true = 0
for index, row in df.iterrows():
   if row['label'] == 'fake':
       fake = fake + 1
   else:
      true = true + 1


# In[9]:


fake,true


# In[10]:


from better_profanity import profanity
dirty_text = "That l3sbi4n did a very good H4ndjob."
profanity.contains_profanity(dirty_text)


# In[11]:


import re
from textblob import TextBlob
import pointofview
from better_profanity import profanity

def get_word_count(row):
    tweet = row[1]
    return len(tweet.split(' '))

def get_character_count(row):
    tweet = row[1]
    return len(tweet)

def get_question_count(row):
    tweet = row[1]
    return tweet.count('?')

def get_exclaimation_count(row):
    tweet = row[1]
    return tweet.count('!')

def has_colon(row):
    tweet = row[1]
    if tweet.find(':') != -1:
        return 1
    else:
        return 0
        
def get_mention_count(row):
    tweet = row[1]
    return tweet.count('@')
    
def get_hashtag_count(row):
    tweet = row[1]
    return tweet.count('#')

def get_url_count(row):
    tweet = row[1]
    return len(re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', tweet))
    
def get_polarity(row):
    tweet = row[1]
    return (TextBlob(tweet)).sentiment[0]

def get_subjectivity(row):
    tweet = row[1]
    return (TextBlob(tweet)).sentiment[1]
    
def get_first_pronouns(row):
    tweet = row[1]
    return len(pointofview.parse_pov_words(tweet)['first'])
    
def get_second_pronouns(row):
    tweet = row[1]
    return len(pointofview.parse_pov_words(tweet)['second'])
    
def get_third_pronouns(row):
    tweet = row[1]
    return len(pointofview.parse_pov_words(tweet)['third'])

def has_profanity(row):
    tweet = row[1]
    if profanity.contains_profanity(tweet):
        return 1
    else:
        return 0

def get_via_count(row):
    tweet = row[1]
    return tweet.lower().count('via')

def get_uppercase_chars(row):
    tweet = row[1]
    return len(re.findall(r'[A-Z]',tweet))


# In[12]:


df['word_count'] = df.apply(lambda row: get_word_count(row), axis=1)
df['character_count'] = df.apply(lambda row: get_character_count(row), axis=1)
df['uppercase_count'] = df.apply(lambda row: get_uppercase_chars(row), axis=1)
df['question_count'] = df.apply(lambda row: get_question_count(row), axis=1)
df['exclaimation_count'] = df.apply(lambda row: get_exclaimation_count(row), axis=1)
df['mention_count'] = df.apply(lambda row: get_mention_count(row), axis=1)
df['hashtag_count'] = df.apply(lambda row: get_hashtag_count(row), axis=1)
df['url_count'] = df.apply(lambda row: get_url_count(row), axis=1)
df['polarity'] = df.apply(lambda row: get_polarity(row), axis=1)
df['subjectivity'] = df.apply(lambda row: get_subjectivity(row), axis=1)
df['first_pronoun_count'] = df.apply(lambda row: get_first_pronouns(row), axis=1)
df['second_pronoun_count'] = df.apply(lambda row: get_second_pronouns(row), axis=1)
df['third_pronoun_count'] = df.apply(lambda row: get_third_pronouns(row), axis=1)
df['via_count'] = df.apply(lambda row: get_via_count(row), axis=1)
df['has_profanity'] = df.apply(lambda row: has_profanity(row), axis=1)
df['has_colon'] = df.apply(lambda row: has_colon(row), axis=1)


# In[13]:


df.drop('tweetId', axis=1, inplace=True)
df.drop('timestamp', axis=1, inplace=True)
df.drop('userId', axis=1, inplace=True)
df.drop('imageId(s)', axis=1, inplace=True)
df.drop('username', axis=1, inplace=True)


# In[14]:


df.head(20)


# In[19]:


import numpy as np
df["label_code"] = np.where(df["label"]=='real', 0, 1)


# In[20]:


df.to_pickle('complete_dataset_features.pkl')


# In[4]:


df = pd.read_pickle("complete_dataset_features.pkl")


# In[5]:


df.groupby('label_code').mean()


# In[6]:


df.mean()


# In[7]:


df_fake = df.loc[(df['label_code'] == 1)]
df_real = df.loc[(df['label_code'] == 0)]


# In[8]:


import seaborn as sns
corr = df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)


# In[9]:


corr = df.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2)
# 'RdBu_r' & 'BrBG' are other good diverging colormaps


# In[10]:


type(corr)


# In[11]:


c1 = corr.abs().unstack().drop_duplicates()
l = c1.sort_values(ascending = False)


# In[12]:


l


# In[13]:


def sort_correlation_matrix(correlation_matrix):
    cor = correlation_matrix.abs()
    top_col = cor[cor.columns[0]][1:]
    top_col = top_col.sort_values(ascending=False)
    ordered_columns = [cor.columns[0]] + top_col.index.tolist()
    return correlation_matrix[ordered_columns].reindex(ordered_columns)


# In[14]:


sort_correlation_matrix(corr)


# In[15]:


corr['label_code'].sort_values()


# In[16]:


df['pronoun_count'] = df['first_pronoun_count'] + df['second_pronoun_count'] + df['third_pronoun_count']


# In[17]:


corr = df.corr()


# In[18]:


corr['label_code'].sort_values()


# In[19]:


import seaborn as sns
corr = df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)


# In[20]:


df['sub_polar'] = df['subjectivity'] + df['polarity']


# In[21]:


corr = df.corr()


# In[22]:


corr['label_code'].sort_values()


# In[23]:


corr = df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)


# In[24]:


df2 = df[['label_code', 'exclaimation_count']]
df2.groupby('label_code').mean()


# In[25]:


df2 = df[['label_code', 'via_count']]
import seaborn as sns
sns.catplot(y='via_count', x='label_code', data=df2, kind='point', aspect=1, )


# In[27]:


df2 = df[['label_code', 'exclaimation_count']]
import seaborn as sns
sns.catplot(y='exclaimation_count', x='label_code', data=df2, kind='point', aspect=1, )


# In[28]:


df2 = df[['label_code', 'question_count']]
import seaborn as sns
sns.catplot(y='question_count', x='label_code', data=df2, kind='point', aspect=1, )


# In[29]:


df2 = df[['label_code', 'has_profanity']]
import seaborn as sns
sns.catplot(y='has_profanity', x='label_code', data=df2, kind='point', aspect=1, )


# In[ ]:




