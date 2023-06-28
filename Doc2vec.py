import math
import re
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import silhouette_score
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import numpy as np
import nltk
import os
import glob

from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA

nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('words')


#Preprocessing

docs = []

for filename in glob.glob(os.path.join('Doc50', '*')):
    with open(filename, 'r') as f:
        content = f.readlines()[10:]  # Leaving out the first 10 lines as they contain metadata
        content = ' '.join(content)  
        docs.append(content)

print(f"Number of documents read: {len(docs)}")

stop = []
with open('Stopword-List.txt', 'r') as f:
    for x in f:
        stop += x.split()


def clean(term):
    ps = nltk.PorterStemmer()
    term = term.lower()
    term = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", term)
    term = re.sub(r'[^\w\s]', '', term)
    term = ps.stem(term)
    return term


tokenized_docs = [nltk.word_tokenize(clean(doc)) for doc in docs if doc not in stop]

# Feature Engineering of Document length, Sentence count, Paragraph count, Named Entity Count

doc_len = [len(doc) for doc in tokenized_docs]
sen_c = [len(nltk.sent_tokenize(doc)) for doc in docs]
par_c = [doc.count('\n\n') + 1 for doc in docs]

ne_c = []
for doc in docs:
    sentences = nltk.sent_tokenize(doc)
    ne = []
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        tag = nltk.pos_tag(words)
        ne += nltk.ne_chunk(tag, binary=True).subtrees(lambda t: t.label() == 'NE')
    ne_c.append(len(ne))

# Baseline: TF-IDF clustering

def tf(t, doc):
    return sum(1 for x in doc if t == x)

def tf_save(tokens):
    tf_docs = {}
    for word in tokens:
        tf_docs[word] = tf_docs.get(word, 0) + 1
    return tf_docs


def idf(t, docs):
    return math.log(len(docs) / (sum(1 for doc in docs if t in doc) + 1))

def idf_save(documents):
    idf_docs = {}
    all_words = set([word for doc in documents for word in doc])
    for word in all_words:
        idf_docs[word] = idf(word, documents)
    return idf_docs



tf_docs = [tf_save(doc) for doc in tokenized_docs]
idf_docs = idf_save(tokenized_docs)


tfidf_docs = []
for tf_dict in tf_docs:
    tfidf_doc = {}
    for word, tf in tf_dict.items():
        tfidf_doc[word] = tf * idf_docs[word]
    tfidf_docs.append(tfidf_doc)


# TF-IDF Weighted Word2Vec
w2v_model = Word2Vec(tokenized_docs, vector_size=200, min_count=20, window=30, sample=1e-3)
word_vectors = np.zeros((len(tfidf_docs), w2v_model.vector_size + 4))  

for i, doc in enumerate(tfidf_docs):
    doc_vector = np.zeros(w2v_model.vector_size + 4)  
    total_weight = 0
    for word, tfidf in doc.items():
        if word in w2v_model.wv:
            word_vector = w2v_model.wv[word] * tfidf
            doc_vector[:-4] += word_vector  
            total_weight += tfidf
        if total_weight > 0:
            doc_vector[:-4] /= total_weight  
    doc_vector[-4] = doc_len[i]  
    doc_vector[-3] = sen_c[i]  
    doc_vector[-2] = par_c[i]  
    doc_vector[-1] = ne_c[i]  
    word_vectors[i] = doc_vector

# Doc2Vec as an extension of Word2Vec------------------------------Doc2vec is used to improve the results
tag_docs = [TaggedDocument(words=doc, tags=[i]) for i, doc in enumerate(tokenized_docs)]
d2v_model = Doc2Vec(tag_docs, vector_size=200, min_count=5, window=30, sample=1e-3)
doc_vectors = np.array([d2v_model.docvecs[i] for i in range(len(tokenized_docs))])
word_vectors = np.concatenate((word_vectors, doc_vectors), axis=1)

# Dimensionality reduction using PCA
pca = PCA(n_components=50)
reduced_vectors = pca.fit_transform(word_vectors)

# K-Means Clustering
km = KMeans(n_clusters=5, n_init=10, max_iter=300)
clusters = km.fit_predict(reduced_vectors)

def top_w(index2word, k, centers, wordvecs):
    tree = KDTree(wordvecs)
    nearest_neighbors = [tree.query(np.reshape(x, (1, -1)), k=min(k, len(wordvecs))) for x in centers]
    nearest__idxs = [x[1] for x in nearest_neighbors]
    nearest_ = {}
    for i in range(0, len(nearest__idxs)):
        nearest_['Cluster #' + str(i)] = [index2word[j] for j in nearest__idxs[i][0]]
    df = pd.DataFrame(nearest_)
    df.index = df.index + 1
    return df


index2word = list(w2v_model.wv.index_to_key)
top_words = top_w(index2word, 10, reduced_vectors, reduced_vectors)
doc_labels = clusters
print("Top Words:")
print(top_words)
for i, label in enumerate(doc_labels):
    print(f"Document {i+1} -> C{label+1}")


subdirectories = next(os.walk('Doc50 GT'))[1]
true = []
for subdir in subdirectories:
    document_files = os.listdir(os.path.join('Doc50 GT', subdir))
    true.extend([subdir] * len(document_files))
for i, label in enumerate(true):
    print(f"Document {i+1} -> True Label: {label}")

# Evaluation by Purity and Silhouette Score
cont_matrix_final = contingency_matrix(doc_labels, true)
purity_final = np.sum(np.amax(cont_matrix_final, axis=0)) / np.sum(cont_matrix_final)
silhouette_avg = silhouette_score(reduced_vectors, doc_labels)
print(f"Purity: {purity_final}")
print(f"Silhouette Score: {silhouette_avg}")
