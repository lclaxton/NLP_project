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

'''
Heres how to tokenise it manually 
# Now to tokenize the words
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')

df.iloc[:,5] = df.iloc[:,5].apply(lambda x: tokenizer.tokenize(x.lower()))

# Now to remove common stopwords
def remove_stopwords(text):
    words = [w for w in text if w not in stopwords.words('english')]
    return words

df.iloc[:,5] = df.iloc[:,5].apply(lambda x: remove_stopwords(x))
'''

from nltk.corpus import stopwords

# Vectorisation
def allinone(text):
    new = text.split("\n")
    new = ' '.join(new)
    new = "".join([c for c in new if c not in string.punctuation])
    new = [w for w in new.split() if w.lower() not in stopwords.words('english')]
    return new
    
    
from sklearn.feature_extraction.text import CountVectorizer

# Might take awhile...
bow_transformer = CountVectorizer(analyzer=allinone).fit(df.iloc[:,5])

# Print total number of vocab words
print(len(bow_transformer.vocabulary_))

bow = bow_transformer.transform([df.iloc[0,5]])
print(bow)
print(bow_transformer.get_feature_names()[95])