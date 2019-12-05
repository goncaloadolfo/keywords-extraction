"""
General auxiliary module.
"""

import numpy as np

TOP_KEYPHRASES_ARGS = 1
THR_KEYPHRASES_ARGS = 2
INVALID_ARGS_EXCEPTION = "invalid number of arguments for getting key phrases (1 or 2)"

PRIORS_UNIFORM = 0
PRIORS_SENTENCE_POS = 1
PRIORS_TFIDF = 2
POSSIBLE_PRIORS = [PRIORS_UNIFORM, PRIORS_SENTENCE_POS, PRIORS_TFIDF]


#####
# key phrases gathering
def candidates_prestige_lists(page_rank_results: dict) -> tuple:
    candidates = np.array(list(page_rank_results.keys()))
    prestige = np.array(list(page_rank_results.values()))
    return candidates, prestige


def get_keyphrases(args: list):
    if len(args) == TOP_KEYPHRASES_ARGS:
        return top_keyphrases(*args)

    elif len(args) == THR_KEYPHRASES_ARGS:
        return thr_keyphrases(*args)

    else:
        raise ValueError(INVALID_ARGS_EXCEPTION)


def thr_keyphrases(page_rank_results: dict, thr: float) -> np.ndarray:
    candidates, prestige = candidates_prestige_lists(page_rank_results)
    return candidates[np.where(prestige > thr)[0]]


def top_keyphrases(page_rank_results: dict, n=5) -> np.ndarray:
    candidates, prestige = candidates_prestige_lists(page_rank_results)
    candidates_copy = candidates.copy()
    prestige_copy = prestige.copy()

    if len(candidates) < n:
        return candidates

    else:
        keyphrases = []

        while len(keyphrases) < n:
            arg_max = np.argmax(prestige_copy)
            keyphrases.append(candidates_copy[arg_max])
            candidates_copy = np.delete(candidates_copy, arg_max)
            prestige_copy = np.delete(prestige_copy, arg_max)

        return keyphrases


#####
# priors calculation
def sentence_pos_priors(candidate_sentence: dict) -> dict:
    sentences_indexes_sum = np.sum(list(candidate_sentence.values()))
    return {candidate: candidate_sentence[candidate] / sentences_indexes_sum
            for candidate in candidate_sentence.keys()}


def tfidf_priors(candidates: np.ndarray, candidates_tfidf: np.ndarray) -> dict:
    tfidf_sum = np.sum(candidates_tfidf)
    return {candidates[i]: candidates_tfidf[i] / tfidf_sum
            for i in range(len(candidates))}


#####
# tfidf array reducing
def reduce_tfidf_array(vocab: np.ndarray, tfidf_array: np.ndarray, candidates: list) -> np.ndarray:
    reduced_array = []

    for candidate in candidates:
        candidate_index = np.where(vocab == candidate)[0][0]
        reduced_array.append(tfidf_array[candidate_index])

    return np.array(reduced_array)
