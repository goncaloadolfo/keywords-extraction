"""
Functions to evaluate keyphrase system evaluation.
"""

import json

import numpy as np

TEST_GT_PATH = "dataset/references/test.uncontr.json"
TRAIN_GT_PATH = "dataset/references/train.uncontr.json"

MEAN_PRECISION_KEY = "mean_precision"
MEAN_RECALL_KEY = "mean_recall"
MEAN_F1_SCORE_KEY = "mean_f1_score"
MEAN_PRECISION5_KEY = "mean_precision_5"
MEAN_AVERAGE_PRECISION_KEY = "mean_average_precision"


def __precision(retrieved_keyphrases, relevant_keyphrases):
    nr_common_keyphrases = len(np.intersect1d(retrieved_keyphrases, relevant_keyphrases))
    # proportion of retrieved keyphrases that are relevant
    return nr_common_keyphrases / len(retrieved_keyphrases)


def __recall(retrieved_keyphrases, relevant_keyphrases):
    nr_common_keyphrases = len(np.intersect1d(retrieved_keyphrases, relevant_keyphrases))
    # proportion of relevant documents retrieved
    return nr_common_keyphrases / len(relevant_keyphrases)


def __f1_score(precision, recall):
    # harmonic mean
    return (2 * precision * recall) / (precision + recall) if precision + recall != 0 else 0.0


def __precision_n(retrieved_keyphrases, relevant_keyphrases, n):
    # precision at n-th retrieved keyphrase
    return __precision(retrieved_keyphrases[:n], relevant_keyphrases)


def __average_precision(retrieved_keyphrases, relevant_keyphrases):
    # average of precision in each recall point
    precisions = [__precision_n(retrieved_keyphrases, relevant_keyphrases, i + 1)
                  for i in range(len(retrieved_keyphrases))
                  if retrieved_keyphrases[i] in relevant_keyphrases]

    return np.sum(precisions) / len(relevant_keyphrases)


def model_evaluation(documents_retrieved_keyphrases, ground_truth):
    results = {}
    doc_ids = documents_retrieved_keyphrases.keys()

    # metrics
    precisions = []
    recalls = []
    f1_scores = []
    precisions_5 = []
    average_precisions = []

    for doc_id in doc_ids:
        # get list of retrieved/ground truth keyphrases
        retrieved_phrases = documents_retrieved_keyphrases[doc_id]
        doc_ground_truth = ground_truth[doc_id]
        flat_ground_truth = [keyphrase for sublist in doc_ground_truth for keyphrase in sublist]

        # calc evaluation metrics
        precision = __precision(retrieved_phrases, flat_ground_truth)
        recall = __recall(retrieved_phrases, flat_ground_truth)
        f1_score = __f1_score(precision, recall)
        precision_5 = __precision_n(retrieved_phrases, flat_ground_truth, 5)
        average_precision = __average_precision(retrieved_phrases, flat_ground_truth)

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)
        precisions_5.append(precision_5)
        average_precisions.append(average_precision)

        # append results of the document to dict
        results[doc_id] = [precision, recall, f1_score]

    # append global mean scores
    mean_precision = np.sum(precisions) / len(doc_ids)
    mean_recall = np.sum(recalls) / len(doc_ids)
    mean_f1_score = np.sum(f1_scores) / len(doc_ids)
    mean_precision_5 = np.sum(precisions_5) / len(doc_ids)
    mean_average_precision = np.sum(average_precisions) / len(doc_ids)

    results[MEAN_PRECISION_KEY] = mean_precision
    results[MEAN_RECALL_KEY] = mean_recall
    results[MEAN_F1_SCORE_KEY] = mean_f1_score
    results[MEAN_PRECISION5_KEY] = mean_precision_5
    results[MEAN_AVERAGE_PRECISION_KEY] = mean_average_precision

    return results


def ground_truth_reader(file_path=TEST_GT_PATH):
    with open(file_path) as json_file:
        return json.load(json_file)
