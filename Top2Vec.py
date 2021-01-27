from sklearn.datasets import fetch_20newsgroups

newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

from top2vec import Top2Vec

model = Top2Vec(documents=newsgroups.data, embedding_model=)