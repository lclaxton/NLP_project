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

# Now to tokenize the words
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')

df.iloc[:,5] = df.iloc[:,5].apply(lambda x: tokenizer.tokenize(x.lower()))


# Now to remove common stopwords
from nltk.corpus import stopwords

def remove_stopwords(text):
    words = [w for w in text if w not in stopwords.words('english')]
    return words

df.iloc[:,5] = df.iloc[:,5].apply(lambda x: remove_stopwords(x))