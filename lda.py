from sklearn.datasets import fetch_20newsgroups
newsgroups_train = fetch_20newsgroups(subset='train', shuffle = True)
newsgroups_test = fetch_20newsgroups(subset='test', shuffle = True)
import gensim
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import numpy as np
np.random.seed(400)



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

for doc in newsgroups_train.data:
    processed_docs.append(preprocess(doc))

dictionary = gensim.corpora.Dictionary(processed_docs)

dictionary.filter_extremes(no_below=15, no_above=0.1, keep_n=100000)

bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

'''
Train your lda model using gensim.models.LdaMulticore and save it to 'lda_model'
'''
lda_model = gensim.models.LdaModel(corpus=bow_corpus, id2word=dictionary, num_topics=20, random_state=100,
   update_every=1, chunksize=100, passes=10, alpha='auto')


for idx, topic in lda_model.print_topics(-1):
    print("Topic: {} \nWords: {}".format(idx, topic ))
    print("\n")


actual_topics = []
predicted_topics = []

y = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
     'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey',
     'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns',
     'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']

x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]


for count, unseen_document in enumerate(newsgroups_test.data):
    # Data preprocessing step for the unseen document
    bow_vector = dictionary.doc2bow(preprocess(unseen_document))

    for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
        predicted_topics.append(index)
        break

    actual_topics.append(newsgroups_test.target[count])



import numpy as np

matrix = np.zeros((20, 20), int)

for i in range(len(predicted_topics)):
 matrix[predicted_topics[i], actual_topics[i]] += 1


print(matrix)
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sn
matrix = pd.DataFrame(matrix, columns = y)
plt.figure(figsize = (10,7))
sn.heatmap(matrix,annot=True, fmt='g')
plt.show()

