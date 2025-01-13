#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cvxpy as cp


# In[7]:


# For data preprocess import
import re
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk


# In[8]:


import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)


# In[9]:


from string import punctuation
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import contractions


# In[10]:


import nltk
nltk.download('punkt')
import nltk
nltk.download('averaged_perceptron_tagger')

from datasets import load_dataset


# In[11]:


import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# # DATA IMPORT

# In[12]:


import pandas as pd


# In[13]:


# Load the large Twitter sentiment dataset
ds = load_dataset("gxb912/large-twitter-tweets-sentiment")

# Convert HuggingFace dataset to pandas DataFrame
twitter_df = pd.DataFrame(ds['train'])

# Print info about the Twitter dataset
print("Twitter Dataset Info:")
print(twitter_df.shape)
print("\nLabel distribution:")
print(twitter_df['sentiment'].value_counts())
print("\nColumns:", twitter_df.columns.tolist())


# In[14]:


# Load our TikTok Shop Twitter dataset
tiktok_df = pd.read_csv(r'F:\Uni Stuff\BUSINESS CASE_GROUP PROJECT\Dataset _tiktokshop.csv' )
print("\nTikTok Dataset Info:")
print(tiktok_df.shape)
print("\nTikTok columns:", tiktok_df.columns.tolist())


# In[15]:


# Create a template DataFrame with all columns from our TikTok dataset
twitter_template = pd.DataFrame(columns=tiktok_df.columns)


# In[16]:


# Prepare Twitter data to match our format
twitter_processed = twitter_template.copy()
twitter_processed['full_text'] = twitter_df['text']
twitter_processed['label'] = twitter_df['sentiment'].map({'negative': 'neg', 'positive': 'pos'})
# Fill other columns with NaN or appropriate default values
twitter_processed['conversation_id_str'] = 'twitter_' + twitter_df.index.astype(str)
twitter_processed['created_at'] = pd.Timestamp.now() 
twitter_processed['favourite_count'] = 0
twitter_processed['id_str'] = 'twitter_' + twitter_df.index.astype(str)
twitter_processed['image_url'] = None 
twitter_processed['in_reply_to_screen_name'] = None  
twitter_processed['lang'] = 'en'  
twitter_processed['location'] = None 
twitter_processed['quote_count'] = 0
twitter_processed['reply_count'] = 0
twitter_processed['retweet_count'] = 0
twitter_processed['tweet_url'] = None 
twitter_processed['user_id_str'] = 'twitter_user_' + twitter_df.index.astype(str)
twitter_processed['username'] = 'twitter_user'

# Add source column
twitter_processed['source'] = 'twitter_dataset'


# In[17]:


# Now combine the datasets
combined_df = pd.concat([tiktok_df, twitter_processed], ignore_index=True)


# In[18]:


# Remove duplicates based on text content
combined_df.drop_duplicates(subset='full_text', inplace=True)


# In[19]:


# Shuffle the dataset
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nCombined Dataset Info:")
print(combined_df.shape)
print("\nLabel distribution:")
print(combined_df['label'].value_counts())


# In[20]:


# Save combined dataset
combined_df.to_csv('combined_twitter_tiktok_dataset.csv', index=False)


# In[21]:


tweets = combined_df
tweets  


# In[22]:


tweets.head()


# In[23]:


tweets_df = tweets[['full_text']]
tweets_df.head()


# In[24]:


tweets_df.shape


# # DATA PREPROCESSING

# In[25]:


tweets.drop_duplicates(inplace = True, subset="full_text")
tweets.duplicated()



# In[26]:


tweets_df


# In[27]:


#To view the information of data
tweets_df.info()


# In[28]:


#To create dataframe
cleaned = tweets_df[["full_text"]]

#Rearrange the index
cleaned_df = cleaned.reset_index(drop = True)
cleaned_df


# In[29]:


#To remove NaN
cleaned_df = cleaned_df.dropna()

#To view the data after removing NaN
cleaned_df


# In[30]:


#Lowercase coversion
cleaned_df['lowercase'] = cleaned_df.full_text.str.lower()
cleaned_df.head()


# In[31]:


#to convert the datatype to string
cleaned_df.full_text = cleaned_df.full_text.astype(str)


# In[32]:


#Remove URL link
# Add this function after the imports
def clean_text(text):
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove numbers and special characters
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text.strip().lower()

# Then apply it
cleaned_df['remove_link'] = cleaned_df['lowercase'].apply(clean_text)
                                                                                                                                              
cleaned_df.head()


# In[33]:


#remove twitter handles (mention)
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
        
        return input_txt


# In[34]:


#remove twitter handles (user)
cleaned_df['mention'] = np.vectorize(remove_pattern)(cleaned_df['remove_link'], "@[\w]*")

cleaned_df.head()


# In[35]:


import string
dir(string)


# In[36]:


string.punctuation


# In[37]:


for char in cleaned_df:
    if char in string.punctuation:
        cleaned_df = cleaned_df.replace(char, '')
        print(cleaned_df)


# In[38]:


def clean_text(full_text):
    for char in full_text:
        full_text = full_text.replace(char, '')
        return full_text


# In[39]:


cleaned_df.head()


# In[40]:


cleaned_df['remove_link'].apply(lambda x: clean_text(x))


# In[41]:


#Remove punctuation
cleaned_df['punctuation'] = cleaned_df.remove_link.apply(lambda x: re.sub(r"[^a-z\s\(\-:\)\\\/\];='#]", '', x))

cleaned_df.head()


# In[42]:


cleaned_df['expand_word'] = cleaned_df['punctuation'].apply(lambda x: [contractions.fix(word) for word in x.split()])


# In[43]:


cleaned_df['expand_word'] = [' '.join(map(str, l)) for l in cleaned_df['expand_word']]

cleaned_df.head()


# In[44]:


#Remove #
cleaned_df['hashtag']= cleaned_df.expand_word.apply(lambda x: re.sub("#[A-Za-z0-9_]+","", x))

cleaned_df.head(3)


# In[45]:


#remove short word that has length word 2 or less
cleaned_df['remove_short'] = cleaned_df['hashtag'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))

cleaned_df.head()


# In[46]:


#tokenization
import nltk
nltk.download('punkt_tab')

cleaned_df['tokenized'] = cleaned_df['remove_short'].apply(word_tokenize)
cleaned_df.head()


# In[47]:


#Stop word library
nltk.download("stopwords")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# In[48]:


#Stopword removal

import nltk
from nltk.corpus import stopwords
set(stopwords.words('english'))

#set of stop words
stop_words = set(stopwords.words('english'))

negation_words = ['not', 'no', 'never', 'none']
for word in negation_words:
    stop_words.discard(word)


cleaned_df['stopwords_removed'] = cleaned_df.tokenized.apply(lambda x: [word for word in x if word not in stop_words])


# In[49]:


cleaned_df.head()


# In[50]:


#pos tag to label the type of word
import nltk
nltk.download('averaged_perceptron_tagger')

cleaned_df['pos_tags'] = cleaned_df['stopwords_removed'].apply(nltk.tag.pos_tag)
cleaned_df.head()


# In[51]:


#To convert pos tag to word net to use lemmatizer
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')


# In[52]:


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
cleaned_df['wordnet_pos'] = cleaned_df['pos_tags'].apply(lambda x: [(word, get_wordnet_pos(pos_tag)) for (word, pos_tag) in x]) 


# In[53]:


cleaned_df.head()


# In[54]:


#Lemmatizer
wnl = WordNetLemmatizer()
cleaned_df['lemmatized'] = cleaned_df['wordnet_pos'].apply(lambda x: [wnl.lemmatize(word, tag) for word, tag in x])
cleaned_df.head()


# In[55]:


#Saving dataframe in csv
cleaned_df.to_csv('Preprocessed_Data')
# tweets_df.to_excel("preprocessed_data.xlsx", index=False)


# # Label Using Text Blob

# In[56]:


#labelling library

import string
import nltk
import plotly.express as px
from nltk.sentiment.util import *


# In[57]:


from wordcloud import WordCloud
import os
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from textblob import TextBlob
from nltk import tokenize


# In[58]:


cleaned_df.head()


# In[59]:


#using polarity
from textblob import TextBlob

# Ensure the 'lemmatized' column is of type string
cleaned_df["lemmatized"] = cleaned_df["lemmatized"].astype("str")
cleaned_df['label'] = ''  # Initialize the 'label' column

# Iterate over the rows of the 'lemmatized' column using items()
for i, x in cleaned_df['lemmatized'].items():
    label = TextBlob(x)
    cleaned_df.at[i, 'label'] = label.sentiment.polarity  # Use .at to set the value
    print("Index: ", i, "label", label.sentiment.polarity)


# In[60]:


cleaned_df.head()


# In[61]:


#to simplify the polarity become positive and negative

def polarity_to_label(x):
    if(x >= -1 and x < 0):
        return 'neg'
    if(x == 0):
        return 'neutral'
    if(x > 0 and x <= 1):
        return 'pos'
cleaned_df.label = cleaned_df.label.apply(polarity_to_label)

cleaned_df.head()


# In[62]:


cleaned_df.label.value_counts()


# In[63]:


#visualise the label

cleaned_df.label.value_counts().plot(kind='bar',title="Sentiment Analysis")


# In[64]:


#removing neutral label since prediction within this project revolve
#with neg and pos only
df = cleaned_df[cleaned_df.label !="neutral"]

df.count()


# In[65]:


df


# In[66]:


#rearrange the index

df = df.reset_index(drop=True)


# In[67]:


df


# # Exploratory Data Analysis
# 

# In[68]:


df


# In[69]:


df.label.value_counts()


# In[70]:


#To check null value

df.lemmatized.isnull().value_counts()


# In[71]:


df.shape


# In[72]:


df.label.value_counts().plot(kind='bar',title="Sentiment Analysis")


# In[73]:


#convert label to numeric values
df['label'] = df.label.map(lambda x: int(1) if x =='pos' 
                           else int(0) if x =='neg' else np.nan)
df.head()


# In[74]:


df= df[['lemmatized','label']]
df.head()


# In[75]:


from sklearn.model_selection import train_test_split
lemmatize = df['lemmatized'].values
labels = df['label'].values
lemmatize_train, lemmatize_test, labels_train, labels_test = train_test_split(lemmatize, labels)

print(lemmatize_train.shape)
print(lemmatize_test.shape)
print(labels_train.shape)
print(labels_test.shape)


# In[76]:


from sklearn.feature_extraction.text import TfidfVectorizer
lemmatizer = WordNetLemmatizer()
corpus=[]
messages = df.lemmatized.tolist()

for i in range(0, len(messages)):
    message = re.sub('[^a-zA-Z]', ' ', str(messages[i]))
    message = message.lower()
    message = message.split()
    message = [lemmatizer.lemmatize(word) for word in message if not word in stopwords.words('english')]
    message = ' '.join(message)
    corpus.append(message)


# In[77]:


vectorizer = TfidfVectorizer(min_df = 5,
                              max_df = 0.8,
                              sublinear_tf = True,
                              use_idf = True)


# In[78]:


tfidf = vectorizer.fit_transform(df.lemmatized)


# In[80]:


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
vectorizer = vectorizer.fit(corpus)
X = vectorizer.transform(corpus)

X.shape


# In[81]:


y = df['label'].replace(0, -1)


# In[82]:


from sklearn.manifold import TSNE
import seaborn as sns
labels = y.to_list()
tsne_results = TSNE(n_components=2,init='random',random_state=0, perplexity=40).fit_transform(X)
plt.figure(figsize=(20,10))
palette = sns.hls_palette(2, l=.3, s=.9)
sns.scatterplot(
    x=tsne_results[:,0], y=tsne_results[:,1],
    hue=labels,
    palette= palette,
    legend="full",
    alpha=0.3
)
plt.show()


# In[83]:


# Split dataset to Testing and Training

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

#70/30
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.30, random_state = 0)


# In[84]:


X_train.shape


# In[85]:


X_test.shape


# # SVM Algorithm

# In[86]:


from collections import Counter


# In[87]:


print("Before SMOTE :" , Counter(y_train))


# In[88]:


sm = SMOTE(random_state=2)
X_train, y_train = sm.fit_resample(X_train, y_train.ravel())


# In[91]:


print("After SMOTE :" , Counter(y_train))


# In[92]:


from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Create and train SVM model with linear kernel
svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train, y_train)


# In[93]:


# Make predictions
predictions = svm_model.predict(X_test)


# In[94]:


# Print results
print("\nClassification Report:")
print(classification_report(y_test, predictions))
print("\nAccuracy Score:", accuracy_score(y_test, predictions))


# In[95]:


from sklearn.metrics import roc_curve, auc
fpr, tpr, _ = roc_curve(y_test, svm_model.decision_function(X_test))
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# In[96]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()


# # WORDCLOUD

# In[98]:


# import required libraries
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd

# read data into a dataframe
df = pd.read_csv(r'F:\Uni Stuff\BUSINESS CASE_GROUP PROJECT\Preprocessed_Data')

# combine all text into a single string
text = ' '.join(df['full_text'].tolist())

# create a WordCloud object
wordcloud = WordCloud(width=800, height=800,
                      background_color='black',
                      stopwords=STOPWORDS,
                      min_font_size=10).generate(text)

# plot the WordCloud image
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)

# show the plot
plt.show()


# In[99]:


# Save the trained model and vectorizer
import pickle
import os

# Create model directory if it doesn't exist
if not os.path.exists('model'):
    os.makedirs('model')

# Save the SVM model
with open('model/svm_model_large.pkl', 'wb') as f:
    pickle.dump(svm_model, f)

# Save the TF-IDF vectorizer
with open('model/vectorizer_large.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Model and vectorizer saved successfully!")

# Optional: Verify the files are saved
if os.path.exists('model/svm_model_large.pkl') and os.path.exists('model/vectorizer_large.pkl'):
    print("Files saved in:")
    print(f"- {os.path.abspath('model/svm_model_large.pkl')}")
    print(f"- {os.path.abspath('model/vectorizer_large.pkl')}")


# In[ ]:





# In[ ]:




