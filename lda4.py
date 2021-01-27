import gensim
from nltk import WordNetLemmatizer, SnowballStemmer
from sklearn.datasets import fetch_20newsgroups
dataset = fetch_20newsgroups(shuffle=True, random_state=1)
documents = dataset.data
targets = dataset.target


def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

stemmer = SnowballStemmer("english")

# Tokenize and lemmatize
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))

    return result

processed_docs = []

for doc in documents:
    processed_docs.append(" ".join(preprocess(doc)))


import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=1000)

tf = tf_vectorizer.fit_transform(processed_docs)

vocab = np.array(tf_vectorizer.get_feature_names())

# convert this to a normal numpy array
dtm = tf.toarray()
# normalize counts to rates per 50 words
rates = 50 * dtm / np.sum(dtm, axis=1, keepdims=True)

# This fills the missing values with zero
rates[np.isnan(rates)] = 0

from sklearn.decomposition import LatentDirichletAllocation

# Train the LDA Model with the documents. Here we need to give the number of topics to cluster on as this is unsupervised learning.
lda = LatentDirichletAllocation(n_components=20,).fit(tf)
def store_topics(model, feature_names, no_top_words):
    """This function is to create a dictionary with the number
    of topics and the corresponding top words."""
    topics = {}
    for topic_idx, topic in enumerate(model.components_):
        print ("Topic", topic_idx)
        print (" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
        topics[topic_idx]=" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])
    return topics


topics = store_topics(lda,vocab,3)


# Run this step to get the corresponding topic proportions for each document
doc_topic = lda.transform(tf)


actual_topics = []
predicted_topics = []

y = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
     'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey',
     'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns',
     'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']

x = [topics.get(i) for i in range(20)]


for i in range(11314):

    topic_most_pr = doc_topic[i].argmax()
    predicted_topics.append(topic_most_pr)
    actual_topics.append(targets[i])
""" print (documents[i])
    print ("--------------------------------")
    print (topic_most_pr)
    print ("--------------------------------")
    print (topics[topic_most_pr])
    print ("--------------------------------")
    print(targets[i])"""



import numpy as np

matrix = np.zeros((20, 20), int)

for i in range(11314):
 matrix[predicted_topics[i], actual_topics[i]] += 1


print(matrix)
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sn
matrix = pd.DataFrame(matrix, columns = y, index=x )
plt.figure(figsize = (12,10))
sn.heatmap(matrix,annot=True, fmt='g')
plt.show()

