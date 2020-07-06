'''
Writing a code to analyse what linkedin requires of entry level data
scientists
'''
import pandas as pd
import numpy as np

df= pd.read_excel('NLP_what_defines_a_data_scientist.xlsx')


# Data cleaning step - ensuring that the qualifications and responsibilities
# columns have no '\n' characters 
for i in np.arange(5,7,1):
    df.iloc[:,i] = df.iloc[:,i].str.split("\n")
    df.iloc[:,i] = df.iloc[:,i].apply(lambda x : ' '.join(x))


# Now to remove punctuation
import string 
def remove_punctuation(text):
    no_punct = "".join([c for c in text if c not in string.punctuation])
    return no_punct
df.iloc[:,5] = df.iloc[:,5].apply(lambda x: remove_punctuation(x.lower()))


# Now to tokenize the words to enable stopword removal
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')
df.iloc[:,5] = df.iloc[:,5].apply(lambda x: tokenizer.tokenize(x.lower()))

# Now to remove common stopwords
from nltk.corpus import stopwords
def remove_stopwords(text):
    words = [w for w in text if w not in stopwords.words('english')]
    return words

df.iloc[:,5] = df.iloc[:,5].apply(lambda x: remove_stopwords(x))

# Now to recombine for analysis
df.iloc[:,5] = df.iloc[:,5].apply(lambda x:" ".join(x))


# Begin vectorisation
from sklearn.feature_extraction.text import CountVectorizer

# Might take awhile...
max_feature_length = 10
bow_transformer = CountVectorizer(max_features=max_feature_length,ngram_range=(1,2)).fit(df.iloc[:,5])

# need to figure out how to get total counts 

# below is counts from one entry 

bow = bow_transformer.transform([df.iloc[0,5]])
print(bow) 

word_list = bow_transformer.get_feature_names()
count_list = bow.toarray().sum(axis=0) 
top_counts = pd.DataFrame(zip(word_list,count_list),columns=['term','count',])
top_counts.sort_values('count',axis=0,inplace=True, ascending=False)
print(top_counts)