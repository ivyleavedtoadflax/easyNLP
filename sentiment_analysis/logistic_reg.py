'''
Data in xml, so need parser: BeautifulSoup

Whilst we could label using the ratings, these data have been pre-classified by the researcher, so we will use this pre-classification as our target variable.

Need two passes over the data to determine the vocabulary size, and which index relates to which word, and then a second pass to create vector representations of the comments.

By using a classifier like logistic regression, it is possible to look at the weights that are being applied to a given world, so we can tell whether it is related to possitive of negative sentiment.
'''
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from bs4 import BeautifulSoup
wordnet_lemmatizer = WordNetLemmatizer()
stopwords = set(w.rstrip() for w in open('stopwords.txt'))

positive_reviews = BeautifulSoup(open('electronics/positive.review').read())
positive_reviews = positive_reviews.findAll('review_text')

negative_reviews = BeautifulSoup(open('electronics/negative.review').read())
negative_reviews = negative_reviews.findAll('review_text')

np.random.shuffle(positive_reviews)
positive_reviews = positive_reviews[:len(negative_reviews)]

def my_tokenizer(s):
    s = s.lower()
    tokens = nltk.tokenize.word_tokenize(s)
    tokens = [t for t in tokens if len(t) > 2]
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]
    tokens = [t fot t in tokens if t not in stopwords]
    return tokens
    pass

word_index_map = {}
current_index = 0

for review in positive_review:
    tokens = my_tokenizer(review.text)
        for token in tokens:
            word_index_map(token) = current_index
            current_index += 1
