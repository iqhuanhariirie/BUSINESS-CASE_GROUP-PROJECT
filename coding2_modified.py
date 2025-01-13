#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cvxpy as cp


# In[2]:


# For data preprocess import
import re
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk


# In[3]:


import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)


# In[4]:


from string import punctuation
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import contractions


# In[5]:


import nltk
nltk.download('punkt')
import nltk
nltk.download('averaged_perceptron_tagger')


# In[6]:


import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# # DATA IMPORT

# In[7]:


import pandas as pd

tweets = pd.read_csv(r'F:\Uni Stuff\BUSINESS CASE_GROUP PROJECT\Dataset _tiktokshop.csv' )
print(tweets)


# In[8]:


tweets  


# In[9]:


tweets.head()


# In[10]:


tweets_df = tweets[['full_text']]
tweets_df.head()


# In[11]:


tweets_df.shape


# # DATA PREPROCESSING

# In[12]:


tweets.drop_duplicates(inplace = True, subset="full_text")
tweets.duplicated()



# In[13]:


tweets_df


# In[14]:


#To view the information of data
tweets_df.info()


# In[15]:


#To create dataframe
cleaned = tweets_df[["full_text"]]

#Rearrange the index
cleaned_df = cleaned.reset_index(drop = True)
cleaned_df


# In[16]:


#To remove NaN
cleaned_df = cleaned_df.dropna()

#To view the data after removing NaN
cleaned_df


# In[17]:


#Lowercase coversion
cleaned_df['lowercase'] = cleaned_df.full_text.str.lower()
cleaned_df.head()


# In[18]:


#to convert the datatype to string
cleaned_df.full_text = cleaned_df.full_text.astype(str)


# In[19]:


#Remove URL link
import re
cleaned_df['remove_link'] = cleaned_df.lowercase.apply(lambda x: re.sub(r'https?:\/\/\S+', '',x))
cleaned_df.lowercase.apply(lambda x: re.sub(r"www\.[a-z]?\.?(com)+|[a-z]+\.(com)", '', x))
                                                                                                                                              
cleaned_df.head()


# In[20]:


#remove twitter handles (mention)
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
        
        return input_txt


# In[21]:


#remove twitter handles (user)
cleaned_df['mention'] = np.vectorize(remove_pattern)(cleaned_df['remove_link'], "@[\w]*")

cleaned_df.head()


# In[22]:


import string
dir(string)


# In[23]:


string.punctuation


# In[24]:


cleaned_df['punctuation'] = cleaned_df['remove_link'].apply(lambda x: ''.join([char for char in x if char not in string.punctuation]))

print(cleaned_df)


# In[25]:


def clean_text(full_text):
    return re.sub(r'[^\w\s]', '', full_text)


# In[26]:


cleaned_df.head()


# In[27]:


cleaned_df['remove_link'].apply(lambda x: clean_text(x))


# In[28]:


#Remove punctuation
cleaned_df['punctuation'] = cleaned_df.remove_link.apply(lambda x: re.sub(r"[^a-z\s\(\-:\)\\\/\];='#]", '', x))

cleaned_df.head()


# In[29]:


cleaned_df['expand_word'] = cleaned_df['punctuation'].apply(lambda x: [contractions.fix(word) for word in x.split()])


# In[30]:


cleaned_df['expand_word'] = [' '.join(map(str, l)) for l in cleaned_df['expand_word']]

cleaned_df.head()


# In[31]:


#Remove #
cleaned_df['hashtag']= cleaned_df.expand_word.apply(lambda x: re.sub("#[A-Za-z0-9_]+","", x))

cleaned_df.head(3)


# In[32]:


#remove short word that has length word 2 or less
cleaned_df['remove_short'] = cleaned_df['hashtag'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))

cleaned_df.head()


# In[33]:


#tokenization
import nltk
nltk.download('punkt_tab')

cleaned_df['tokenized'] = cleaned_df['remove_short'].apply(word_tokenize)
cleaned_df.head()


# In[34]:


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


# In[35]:


#Stopword removal

import nltk
from nltk.corpus import stopwords
set(stopwords.words('english'))

#set of stop words
stop_words = set(stopwords.words('english'))

cleaned_df['stopwords_removed'] = cleaned_df.tokenized.apply(lambda x: [word for word in x if word not in stop_words])


# In[36]:


cleaned_df.head()


# In[37]:


#pos tag to label the type of word
import nltk
nltk.download('averaged_perceptron_tagger_eng')

cleaned_df['pos_tags'] = cleaned_df['stopwords_removed'].apply(nltk.tag.pos_tag)
cleaned_df.head()


# In[38]:


#To convert pos tag to word net to use lemmatizer
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')


# In[39]:


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


# In[40]:


cleaned_df.head()


# In[41]:


#Lemmatizer
wnl = WordNetLemmatizer()
cleaned_df['lemmatized'] = cleaned_df['wordnet_pos'].apply(lambda x: [wnl.lemmatize(word, tag) for word, tag in x])
cleaned_df.head()


# In[42]:


#Saving dataframe in csv
cleaned_df.to_csv('Preprocessed_Data')
# tweets_df.to_excel("preprocessed_data.xlsx", index=False)


# # Label Using Text Blob

# In[43]:


#labelling library

import string
import nltk
import plotly.express as px
from nltk.sentiment.util import *


# In[44]:


from wordcloud import WordCloud
import os
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from textblob import TextBlob
from nltk import tokenize


# In[45]:


cleaned_df.head()


# In[46]:


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


# In[47]:


cleaned_df.head()


# In[48]:


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


# In[49]:


cleaned_df.label.value_counts()


# In[50]:


#visualise the label

cleaned_df.label.value_counts().plot(kind='bar',title="Sentiment Analysis")


# In[51]:


#removing neutral label since prediction within this project revolve
#with neg and pos only
df = cleaned_df[cleaned_df.label !="neutral"]

df.count()


# In[52]:


df


# In[53]:


#rearrange the index

df = df.reset_index(drop=True)


# In[54]:


df


# # Exploratory Data Analysis
# 

# In[55]:


df


# In[56]:


df.label.value_counts()


# In[57]:


#To check null value

df.lemmatized.isnull().value_counts()


# In[58]:


df.shape


# In[59]:


df.label.value_counts().plot(kind='bar',title="Sentiment Analysis")


# In[60]:


#convert label to numeric values
df['label'] = df.label.map(lambda x: int(1) if x =='pos' 
                           else int(0) if x =='neg' else np.nan)
df.head()


# In[61]:


df= df[['lemmatized','label']]
df.head()


# In[62]:


from sklearn.model_selection import train_test_split
lemmatize = df['lemmatized'].values
labels = df['label'].values
lemmatize_train, lemmatize_test, labels_train, labels_test = train_test_split(lemmatize, labels)

print(lemmatize_train.shape)
print(lemmatize_test.shape)
print(labels_train.shape)
print(labels_test.shape)


# In[63]:


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


# In[64]:


vectorizer = TfidfVectorizer(min_df = 5,
                              max_df = 0.8,
                              sublinear_tf = True,
                              use_idf = True)


# In[65]:


tfidf = vectorizer.fit_transform(df.lemmatized)


# In[66]:


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
vectorizer = vectorizer.fit(corpus)
X = vectorizer.transform(corpus).toarray()

X.shape


# In[67]:


y = df['label'].replace(0, -1)


# In[68]:


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


# In[69]:


# Split dataset to Testing and Training

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

#70/30
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.30, random_state = 0)


# In[70]:


X_train.shape


# In[71]:


X_test.shape


# # SVM Algorithm

# In[72]:


from collections import Counter


# In[73]:


print("Before SMOTE :" , Counter(y_train))


# In[74]:


sm = SMOTE(random_state=2, sampling_strategy='auto', k_neighbors=5)
X_train_balanced, y_train_balanced = sm.fit_resample(X_train, y_train)


# In[75]:


print("After SMOTE :" , Counter(y_train))


# In[76]:


#To create kernel Matrix
k_value = np.array(X_train @ X_train.T + np.identity(len(y_train))*1e-12)
k_value


# In[77]:


#To set up and minimize the dual function
#Create optimization variables
alpha = cp.Variable(shape = y_train.shape)

#To simplify notation
beta = cp.multiply(alpha, y_train)

K = cp.Parameter(shape = k_value.shape, PSD = True, value = k_value)

#Objective function
obj = .5 * cp.quad_form(beta, K) - np.ones(alpha.shape).T @ alpha

class_weights = dict(zip(np.unique(y_train), 
                        [1 / (np.sum(y_train == c) / len(y_train)) for c in np.unique(y_train)]))

#Constraints
const = [np.array(y_train.T) @ alpha == 0,
         -alpha <= np.zeros(alpha.shape),
         alpha <= [10 * class_weights[y] for y in y_train]]

prob = cp.Problem(cp.Minimize(obj), const)


# In[78]:


result = prob.solve()


# In[79]:


np.linalg.cholesky(k_value) #to check PSD


# In[80]:


w = np.multiply(y_train, alpha.value).T @ X_train


# In[81]:


S = (alpha.value > 1e-4).flatten()
b = y_train[S] - X_train[S] @ w
# b = b[0]
b = np.mean(b)


# In[82]:


def classify(x):
    result = w @ x + b
    return np.sign(result)


# In[83]:


import numpy as np
correct = 0
incorrect = 0
predictions = []
for i in X_test:
    my_svm = classify(i)
    
    predictions = np.append(predictions, my_svm)
    
    predictions = np.array(predictions)


# In[84]:


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
print(classification_report(y_test,predictions))


# In[85]:


accuracy_score(y_test, predictions)
print('Model accuracy score with linear kernel and C=100.0 : {0:0.4f}'. format(accuracy_score(y_test, predictions)*100))


# In[86]:


import sklearn.metrics as metrics
# calculate the fpr and tpr for all thresholds of the classification
fpr, tpr, threshold = metrics.roc_curve(y_test, predictions)
roc_auc = metrics.auc(fpr, tpr)


# In[87]:


import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[88]:


# import matplotlib.pyplot as plt
# import numpy
# from sklearn import metrics

# actual = numpy.random.binomial(1,.9,size = 1000)
# predicted = numpy.random.binomial(1,.9,size = 1000)

# confusion_matrix = metrics.confusion_matrix(actual, predicted)

# cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

# cm_display.plot()
# plt.show()

import matplotlib.pyplot as plt
from sklearn import metrics

# Create and display confusion matrix
confusion_matrix = metrics.confusion_matrix(y_test, predictions)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, 
                                          display_labels=['Negative', 'Positive'])

plt.figure(figsize=(8, 6))
cm_display.plot()
plt.title('Confusion Matrix')
plt.show()


# # WORDCLOUD

# In[90]:


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


# In[ ]:





# In[ ]:





# In[ ]:




