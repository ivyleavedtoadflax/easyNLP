'''
Stemming and lemmatization
'''

# Stemming example

from nltk import PorterStemmer

porter_stemmer = PorterStemmer()

print(porter_stemmer.stem('wolves'))

# Lemmatizing example

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize('wolves'))
