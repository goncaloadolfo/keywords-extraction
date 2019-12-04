"""
Exercise 2 - Improvement of graph ranking; evaluation of
different approaches on a data set.
"""

import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer

from doc_reader import read_dataset_from_pickle, DATASET_PICKLE_PATH
from evaluation_functions import *
from graph_constructor import GraphBuilder, CO_OCCURRENCES_WEIGHTS
from utils import sentence_pos_priors, tfidf_priors, reduce_tfidf_array, get_keyphrases, PRIORS_UNIFORM, \
    PRIORS_SENTENCE_POS

###
# system parameters
n = 1  # 1 <= n <= 3
weight_method = CO_OCCURRENCES_WEIGHTS  # {UNITARY_WEIGHTS, CO_OCCURRENCES_WEIGHTS, CANDIDATE_SIMILARITY_WEIGHTS}
priors_method = PRIORS_UNIFORM  # {PRIORS_UNIFORM, PRIORS_SENTENCE_POS, PRIORS_TFIDF}
###

# load documents
train_docs, test_docs = read_dataset_from_pickle("../" + DATASET_PICKLE_PATH)
all_docs = {**train_docs, **test_docs}
doc_ids = list(all_docs.keys())
collection = list(all_docs.values())

# tfidf and vocabulary
tfidf_obj = TfidfVectorizer(stop_words="english", ngram_range=(1, n))
tfidf_array = tfidf_obj.fit_transform(collection).toarray()
vocab = np.array(tfidf_obj.get_feature_names())

# evaluate
results = {}

for doc_id_index in range(len(doc_ids)):
    print("document {}/{}".format(doc_id_index, len(all_docs)))
    doc_id = doc_ids[doc_id_index]
    doc = all_docs[doc_id]
    candidates = tfidf_obj.fit([doc]).get_feature_names()

    # build graph
    graph_obj = GraphBuilder(doc, collection, candidates, weight_method, n)
    doc_graph = graph_obj.build_graph()

    # page rank
    if priors_method == PRIORS_UNIFORM:
        final_prestige = nx.pagerank(doc_graph, max_iter=50, weight='weight')

    elif priors_method == PRIORS_SENTENCE_POS:
        priors = sentence_pos_priors(graph_obj.get_candidates_sentences())
        final_prestige = nx.pagerank(doc_graph, personalization=priors, max_iter=100, weight='weight')

    else:
        priors = tfidf_priors(candidates, reduce_tfidf_array(vocab, tfidf_array[doc_id_index], candidates))
        final_prestige = nx.pagerank(doc_graph, personalization=priors, max_iter=50, weight='weight')

    # top 5 key phrases
    keyphrases = get_keyphrases([final_prestige])
    results[doc_id] = keyphrases

# evaluation
train_gt = ground_truth_reader("../" + TRAIN_GT_PATH)
test_gt = ground_truth_reader("../" + TEST_GT_PATH)
gt = {**train_gt, **test_gt}
evaluation_results = model_evaluation(results, gt)

print("precision: ", evaluation_results[MEAN_PRECISION_KEY])
print("recall: ", evaluation_results[MEAN_RECALL_KEY])
print("mean average precision: ", evaluation_results[MEAN_AVERAGE_PRECISION_KEY])
