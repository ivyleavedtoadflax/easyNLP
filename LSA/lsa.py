import numpy as np
import nltk
import matplotlib.pyplot as plot
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import TruncatedSVD

wornet_lemmatizer = WordNetLemmatizer()

titles = [line.rstrip() for line in open('all_book_titles.txt')]
stopwords = set(w.rstrip() for w in open('stopwords.txt'))

stopwords = stopwords.union({
        'introduction', 'edition', 'series', 'application',
            'approach', 'card', 'access', 'package', 'plus', 'etext',
                'brief', 'vol', 'fundamental', 'guide', 'essential', 'printed',
                    'third', 'second', 'fourth', })


def my_tokenizer(s):
    s = s.lower()
    tokens = nltk.tokenize.word_tokenize(s)
    tokens = [t for t in tokens if len(t) > 2]
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]
    tokens = [t for t in tokens if t not in stopwords]
    tokens = [t for t in tokens if not any(c.isdigit() for c in t)]
    return tokens
    pass

word_index_map = {}
current_index = 0
all_tokens = []
all_titles = []
index_word_map = []

for title in titles:
    try:
        title = title.encode('ascii','ignore')
        all_titles.append(title) 
        tokens = my_tokenizer(title)
        all_tokens.append(tokens)
        for token in tokens:
        	if token not in word_index_map:
        		word_index_map[token] = current_index
        		current_index += 1
        		index_word_map.append(token)
    except:
    	pass

def tokens_to_vector(tokens):
	x = np.zeros(len(word_index_map))
	for t in tokens:
		i = word_index_map[t]
		x[i] = 1
	return(x)

N = len(all_tokens)
D = len(word_index_map)
X = np.zeros((D, N))
i = 0

for tokens in all_tokens:
	X[:,i] = tokens_to_vector(tokens)
	i += 1

svd = TruncatedSVD()
Z = svd.fit_transform(X)	
