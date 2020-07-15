'''
Writing a code to analyse what linkedin requires of entry level data
scientists
'''
import pandas as pd
import numpy as np

df= pd.read_excel('NLP_what_defines_a_data_scientist.xlsx')


# Data cleaning step - ensuring that the qualifications and responsibilities
# columns have no '\n' characters 
for i in np.arange(7,9,1):
    df.iloc[:,i] = df.iloc[:,i].str.split("\n")
    df.iloc[:,i] = df.iloc[:,i].apply(lambda x : ' '.join(x))


# Now to remove punctuation
import string 
def remove_punctuation(text):
    no_punct = "".join([c for c in text if c not in string.punctuation])
    return no_punct
df.iloc[:,7] = df.iloc[:,7].apply(lambda x: remove_punctuation(x.lower()))
df.iloc[:,8] = df.iloc[:,8].apply(lambda x: remove_punctuation(x.lower()))

# Now to tokenize the words to enable stopword removal
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')
df.iloc[:,7] = df.iloc[:,7].apply(lambda x: tokenizer.tokenize(x.lower()))
df.iloc[:,8] = df.iloc[:,8].apply(lambda x: tokenizer.tokenize(x.lower()))

specific_words = ['experience','ability','skills','strong','etc']

# Now to remove common stopwords
from nltk.corpus import stopwords
def remove_stopwords(text):
    english_words = [w for w in text if w not in stopwords.words('english')]
    additional_words = [w for w in english_words if w not in specific_words]
    return additional_words

df.iloc[:,7] = df.iloc[:,7].apply(lambda x: remove_stopwords(x))
df.iloc[:,8] = df.iloc[:,8].apply(lambda x: remove_stopwords(x))

# Now to recombine for analysis
df.iloc[:,7] = df.iloc[:,7].apply(lambda x:" ".join(x))
df.iloc[:,8] = df.iloc[:,8].apply(lambda x:" ".join(x))

# Begin vectorisation
from sklearn.feature_extraction.text import CountVectorizer

# Might take awhile...
max_feature_length = 10

list_of_transformers=[]
top_word_dataframes = []

for i in np.arange(1,4,1):
    bow_transformer = CountVectorizer(max_features=max_feature_length,ngram_range=(i,i)).fit(df.iloc[:,7])
    bow = bow_transformer.transform([' '.join(df.iloc[:,7].values)])
    list_of_transformers.append(bow)
    word_list = bow_transformer.get_feature_names()
    count_list = bow.toarray().sum(axis=0) 
    top_counts = pd.DataFrame(zip(word_list,count_list),columns=['term','count',])
    top_counts.sort_values('count',axis=0,inplace=True, ascending=False)
    top_word_dataframes.append(top_counts)
    
print(top_word_dataframes)


list_of_transformers_responsibilites =[]
top_word_responsibilities = []

for i in np.arange(1,4,1):
    bow_transformer = CountVectorizer(max_features=max_feature_length,ngram_range=(i,i)).fit(df.iloc[:,8])
    bow = bow_transformer.transform([' '.join(df.iloc[:,8].values)])
    list_of_transformers_responsibilites.append(bow)
    word_list = bow_transformer.get_feature_names()
    count_list = bow.toarray().sum(axis=0) 
    top_counts = pd.DataFrame(zip(word_list,count_list),columns=['term','count',])
    top_counts.sort_values('count',axis=0,inplace=True, ascending=False)
    top_word_responsibilities.append(top_counts)

print(top_word_responsibilities)
# To do:
    # Calculate term frequency 
    # Make some plots?
    # Add in a machine learning element
        # Add new column for would I like this job
        # get model to predict whether I would like that job or not or should I apply
        