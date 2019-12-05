"""
Rank aggregation approach.
Graph formulation and page rank config: uniform priors and uniform weights.
"""

import pickle
import time

import numpy as np
from networkx import pagerank
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from doc_reader import read_dataset_from_pickle, read_doc_from_disk, DATASET_PICKLE_PATH, TEST_DOCUMENT_PATH
from evaluation_functions import model_evaluation, ground_truth_reader, \
    MEAN_PRECISION_KEY, MEAN_RECALL_KEY, MEAN_AVERAGE_PRECISION_KEY, TRAIN_GT_PATH, TEST_GT_PATH
from graph_constructor import GraphBuilder, UNITARY_WEIGHTS
from utils import get_keyphrases

CANDIDATES_VECTORS_PICKLE_PATH = "../files/candidates_vectors.p"
VOCAB_KEY = "vocab"
VECTORS_KEY = "vectors"
n = 1


def candidate_vectors(document: str, background_collection: list) -> tuple:
    # get candidates and tf
    counter = CountVectorizer(stop_words="english", ngram_range=(1, n))
    tf = counter.fit_transform([document]).toarray()[0]
    vocab = counter.get_feature_names()
    tf = tf / len(vocab)

    # idf and tfidf
    tfidf_obj = TfidfVectorizer(stop_words="english", vocabulary=vocab)
    all_docs = [document] + background_collection
    tfidf = tfidf_obj.fit_transform(all_docs).toarray()[0]
    idf = tfidf_obj.idf_

    # centrality
    doc_graph = GraphBuilder(document, background_collection, vocab, UNITARY_WEIGHTS, n).build_graph()
    centrality_dict = pagerank(doc_graph, max_iter=50, weight='weight')

    # vector for each candidate
    vectors = []
    for vocab_i in range(len(vocab)):
        tf_value = tf[vocab_i]
        idf_value = idf[vocab_i]
        tfidf_value = tfidf[vocab_i]
        centrality = centrality_dict[vocab[vocab_i]]
        centrality = centrality if type(centrality) == float else centrality[0][0]
        vectors.append([tf_value, idf_value, tfidf_value, centrality])

    return vocab, vectors


def save_dataset_vectors(pickle_path: str) -> None:
    train_docs, test_docs = read_dataset_from_pickle("../" + DATASET_PICKLE_PATH)
    all_docs = {**train_docs, **test_docs}

    doc_ids = list(all_docs.keys())
    all_docs_list = list(all_docs.values())

    d = {}
    i = 1
    for doc_id in doc_ids:
        print("calculating vectors to doc {}/{}".format(i, len(doc_ids)))
        i += 1
        document = all_docs[doc_id]
        vocab, vectors = candidate_vectors(document, all_docs_list)
        d[doc_id] = {VOCAB_KEY: vocab, VECTORS_KEY: np.array(vectors)}

    with open(pickle_path, "wb") as file:
        pickle.dump(d, file, protocol=pickle.HIGHEST_PROTOCOL)


def doc_keyphrases(vocab: np.ndarray, scores: np.ndarray) -> list:
    scores_per_feature = 1.0 / (50.0 + scores)
    scores_per_feature = np.sum(scores_per_feature, axis=1)
    scores_dict = {vocab[i]: scores_per_feature[i] for i in range(len(vocab))}
    return get_keyphrases([scores_dict])


def simple_document_test(document_path: str) -> list:
    document = read_doc_from_disk(document_path)
    train_docs, test_docs = read_dataset_from_pickle("../" + DATASET_PICKLE_PATH)
    background_collection = list({**train_docs, **test_docs}.values())
    vocab, candid_vectors = candidate_vectors(document, background_collection)
    kps = doc_keyphrases(vocab, np.array(candid_vectors))

    print("Document: ", document)
    print("Candidates: ", vocab)
    print("vectors: ", candid_vectors)
    print("key phrases: ", kps)

    return kps


def main():
    # save_dataset_vectors(CANDIDATES_VECTORS_PICKLE_PATH)

    simple_document_test("../" + TEST_DOCUMENT_PATH)

    t0 = time.time()
    v: dict
    with open(CANDIDATES_VECTORS_PICKLE_PATH, "rb") as file:
        v = pickle.load(file)

    docs_ids = list(v.keys())
    results = {}

    # get keyphrases for each document
    for doc_id_i in range(len(docs_ids)):
        print("Getting key phrases for document {}/{}".format(doc_id_i, len(docs_ids)))
        doc_id = docs_ids[doc_id_i]
        doc_dict = v[doc_id]

        doc_vocab = doc_dict[VOCAB_KEY]
        doc_vectors = doc_dict[VECTORS_KEY]
        doc_kps = doc_keyphrases(doc_vocab, doc_vectors)
        results[doc_id] = doc_kps

    # evaluation results
    train_gt = ground_truth_reader("../" + TRAIN_GT_PATH)
    test_gt = ground_truth_reader("../" + TEST_GT_PATH)
    gt = {**train_gt, **test_gt}
    evaluation_results = model_evaluation(results, gt, soft=True)
    timestamp = time.time() - t0

    print("precision: ", evaluation_results[MEAN_PRECISION_KEY])
    print("recall: ", evaluation_results[MEAN_RECALL_KEY])
    print("mean average precision: ", evaluation_results[MEAN_AVERAGE_PRECISION_KEY])
    print("processing time: ", timestamp)


if __name__ == "__main__":
    main()
