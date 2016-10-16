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

# Remove stopwords based on a list (could use nltk stopwords here)

stopwords = set(w.rstrip() for w in open('stopwords.txt'))

# Load positive and negative reviews

positive_reviews = BeautifulSoup(open('electronics/positive.review').read())
positive_reviews = positive_reviews.findAll('review_text')

negative_reviews = BeautifulSoup(open('electronics/negative.review').read())
negative_reviews = negative_reviews.findAll('review_text')

# Do a random inplace shuffle of the rows

np.random.shuffle(positive_reviews)

# To avoid imbalance, truncate the number of positive rows based on the
# number of negative reviews

positive_reviews = positive_reviews[:len(negative_reviews)]

# Define a custom tokeniser which:
# Convert to lowercase
# Only tokenises words with more than 2 characters
# Uses wordnet lemmatization
# strip stopwords

def my_tokenizer(s):
    s = s.lower()
    tokens = nltk.tokenize.word_tokenize(s)
    tokens = [t for t in tokens if len(t) > 2]
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]
    tokens = [t for t in tokens if t not in stopwords]
    return tokens
    pass

# Create mapping to indices for words

word_index_map = {}
current_index = 0

# Create lists to save the tokenized words to

positive_tokenized = []
negative_tokenized = []

# Loop through for positive reviews
# Apply tokenizer to each word in the review.
# Append the token to the tokenizer list.
# Create the index map:
#   If the word is not already in the map
#   Assign the index for that token to the current_index (the accumulator)
#   Advance the accumulator for the next word

for review in positive_reviews:
    tokens = my_tokenizer(review.text)
    positive_tokenized.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1

# Loop through negative reviews and do the same

for review in negative_reviews:
    tokens = my_tokenizer(review.text)
    negative_tokenized.append(tokens)
    for token in tokens: 
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1

'''
Now create a data array for each token. Each list within the list 
positive_tokenized contains all the tokenised words from the original
review.
'''

def tokens_to_vector(tokens, label):

    # Create an array of the same length +1 as the word_index_map

    x = np.zeros(len(word_index_map) + 1)

    # Loop through the tokens

    for t in tokens:

        # Get the token index from word_index_map

        i = word_index_map[t]

        # Increment the array +1 for the presence of each token

        x[i] += 1

    # Get the proportion of each word based on the total number of words in the corpus

    x = x / x.sum()

    # Apply the label (is it positive or negative)

    x[-1] = label
    return x


# Calculate total length

N = len(positive_tokenized) + len(negative_tokenized)
    
# Instantiate  an array which will become the TDM.

data = np.zeros((N, len(word_index_map) + 1))
i = 0

# positive_tokenized now contains a list of lists of all the tokens
# from each of the positive reviews. We need these to be converted 
# into the TDM.

for tokens in positive_tokenized:
    xy = tokens_to_vector(tokens, 1)
    data[i,:] = xy
    i += 1

for tokens in negative_tokenized:
    xy = tokens_to_vector(tokens, 0)
    data[i,:] = xy
    i += 1

# data is now a 2000 row (document), 11092 column (term) matrix

# Shuffle again for training

np.random.shuffle(data)

# Setup training and test sets

X = data[:,:-1]
Y = data[:, -1]

Xtrain = X[:-100,]
Ytrain = Y[:-100,]

Xtest = X[-100:,]
Ytest = Y[-100:,]

model = LogisticRegression()
model.fit(Xtrain,Ytrain)
print(model.score(Xtest, Ytest))

# Model classification rates are not great, so look at the weights related
# to each term which is over or under a treshold.

threshold = 0.5

for word, index in word_index_map.items():
        weight = model.coef_[0][index]
        if weight > threshold or weight < -threshold:
            print(word, weight)
