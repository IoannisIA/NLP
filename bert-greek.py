from flair.embeddings import TransformerDocumentEmbeddings
from flair.data import Sentence
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
embedding = TransformerDocumentEmbeddings('nlpaueb/bert-base-greek-uncased-v1', layers='-1, -2, -3, -4')
import unicodedata
from scipy.optimize import linear_sum_assignment as linear_assignment
import numpy as np
import re
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def acc(ypred, y):
    """
    Calculating the clustering accuracy. The predicted result must have the same number of clusters as the ground truth.

    ypred: 1-D numpy vector, predicted labels
    y: 1-D numpy vector, ground truth
    The problem of finding the best permutation to calculate the clustering accuracy is a linear assignment problem.
    This function construct a N-by-N cost matrix, then pass it to scipy.optimize.linear_sum_assignment to solve the assignment problem.

    """
    assert len(y) > 0
    assert len(np.unique(ypred)) == len(np.unique(y))

    s = np.unique(ypred)
    t = np.unique(y)

    N = len(np.unique(ypred))
    C = np.zeros((N, N), dtype=np.int32)
    for i in range(N):
        for j in range(N):
            idx = np.logical_and(ypred == s[i], y == t[j])
            C[i][j] = np.count_nonzero(idx)

    # convert the C matrix to the 'true' cost
    Cmax = np.amax(C)
    C = Cmax - C
    #
    indices = linear_assignment(C)
    indices = np.asarray(indices)
    indices = np.transpose(indices)
    row = indices[:][:, 0]
    col = indices[:][:, 1]
    # calculating the accuracy according to the optimal assignment
    count = 0
    for i in range(N):
        idx = np.logical_and(ypred == s[row[i]], y == t[col[i]])
        count += np.count_nonzero(idx)

    return 1.0 * count / len(y)


def text_cleaner(s):
    s = re.sub('[a-zA-Z]+', ' ', s)
    s = re.sub('\d', ' ', s)
    s = re.sub('[!()’\-\[\]«»“{};:€\'",+=<>/?@#$|%^&\n\t—*–_~…]', ' ', s)
    s = re.sub('\.{2,10}', ' ', s)
    s = re.sub('[\s\.]{3,}', ' ', s)
    s = re.sub('( \. )+', ' ', s)
    s = re.sub('(\.){2,}', ' ', s)
    s = re.sub(' [α-ω]{1,3}(\.) ', ' ', s)
    s = re.sub(' [α-ω]\.[α-ω]\. ', ' ', s)
    s = re.sub(' [α-ω]\.[α-ω]\.[α-ω]\. ', ' ', s)
    s = re.sub(' +', ' ', s)
    s = re.sub('^ ', '', s)
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn').lower()
    return s


def make_embeddings(docs, actual_labels, num_sentences):

    print("Splitting each doc to sentences and making embeddings ...")
    docs_embeddings = []
    indexes_for_removal = []
    for doc in tqdm(docs):
        num_sent = num_sentences
        splitted_sentences_of_each_doc = []
        for sentence in doc.split("."):
            if num_sent == 0:
                break
            if sentence != "" and 350 > len(sentence.split()) > 3:
                num_sent -= 1
                splitted_sentences_of_each_doc.append(Sentence(sentence))
        embedding.embed(splitted_sentences_of_each_doc)
        embedding_array_for_all_sentences = []
        for sentence in splitted_sentences_of_each_doc:
            embedding_array_for_all_sentences.append(sentence.embedding.tolist())
            sentence.clear_embeddings()
        embedding_array_for_all_sentences = np.matrix(embedding_array_for_all_sentences).mean(0).tolist()
        if len(embedding_array_for_all_sentences[0]) > 0:
            docs_embeddings.append(embedding_array_for_all_sentences[0])
        else:
            print("Doc embedding zero, removing from docs...")
            index = docs.index(doc)
            indexes_for_removal.append(index)

    for index in indexes_for_removal:
        docs.pop(index)
        actual_labels.pop(index)

    return docs, actual_labels, docs_embeddings


def clustering(docs_embeddings, actual_labels):
    print("Clustering...")
    num_cl = len(np.unique(actual_labels))
    cluster = KMeans(n_clusters=num_cl, init='k-means++', n_init=10, max_iter=300, tol=0.0001, verbose=0,
                     random_state=None, copy_x=True, algorithm='auto').fit(docs_embeddings)
    return cluster.labels_


def getKeysByValue(dictOfElements, valueToFind):
    listOfKeys = list()
    listOfItems = dictOfElements.items()
    for item  in listOfItems:
        if item[1] == valueToFind:
            listOfKeys.append(item[0])
    return  listOfKeys


def print_clusters(docs, predicted_labels):
    print("Making results readable...")
    dictionary = dict(zip([doc for doc in docs], predicted_labels))
    for n in range(len(set(predicted_labels))):
        print("cluster:"+str(n), getKeysByValue(dictionary, n))


def pca_tsne(docs_embeddings, pca_dim, tsne_dim):
    print("PCA...")
    pca = PCA(n_components=pca_dim)
    principalComponents = pca.fit_transform(docs_embeddings)

    print("TSNE...")
    tsne = TSNE(n_components=tsne_dim)
    return tsne.fit_transform(principalComponents)


def plot_clusters(predicted_labels, tsne_results, text):

    """
    print("Calculating and plotting cosine similarity...")
    plt.imshow(cosine_similarity(docs_embeddings))
    plt.colorbar()
    """

    print("Plotting clusters...")
    principalDf = pd.DataFrame(data=tsne_results, columns = ['X', 'Y'])
    finalDf = pd.concat([principalDf, pd.DataFrame(data=predicted_labels, columns=['target'])], axis=1)
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('X', fontsize=15)
    ax.set_ylabel('Y', fontsize=15)
    ax.set_title(text, fontsize=20)
    for target in predicted_labels:
        indicesToKeep = finalDf['target'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'X']
                   , finalDf.loc[indicesToKeep, 'Y']
                   , s = 50)
    ax.legend(set(predicted_labels))
    ax.grid()
    plt.show()


def read_and_clean_dataset_from_file(file_name):
    print("Reading and clearing dataset...")
    data = pd.read_csv(file_name, error_bad_lines=False, delimiter=';',
                       names=['cluster_id', 'title', 'text']).dropna()

    docs = [text_cleaner(doc) for doc in data['text'].values.tolist()]

    actual_labels = [actual_label for actual_label in data['cluster_id'].values.tolist()]

    titles = [text_cleaner(title) for title in data['title'].values.tolist()]

    docs = [i + "." + j for i, j in zip(titles, docs)]

    #docs = [i + "." + j for i, j in zip(titles, [''] * len(titles))]

    #docs = [i + "." + j for i, j in zip([''] * len(docs), docs)]

    return docs, actual_labels




accuracy = []
docs, actual_labels = read_and_clean_dataset_from_file('googlenews_text_from_links.csv')
for i in range(1, 30):
    docs, actual_labels, docs_embeddings = make_embeddings(docs, actual_labels, i)
    predicted_labels = clustering(docs_embeddings, actual_labels)
    #print_clusters(docs, predicted_labels)
    accuracy_tmp = acc(list(predicted_labels), actual_labels)
    accuracy.append(accuracy_tmp)
    tsne_results = pca_tsne(docs_embeddings, 50, 2)
    plot_clusters(predicted_labels, tsne_results,
                  'First '+str(i)+' sentence(s) - accuracy: '
                  +str(round(accuracy_tmp, 2)))


print(accuracy)
