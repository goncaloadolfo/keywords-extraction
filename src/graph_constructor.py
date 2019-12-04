"""
Document to graph conversion.
"""

import pickle

import networkx as nx
import nltk
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer

UNITARY_WEIGHTS = 0
CO_OCCURRENCES_WEIGHTS = 1
CANDIDATE_SIMILARITY_WEIGHTS = 2
POSSIBLE_WEIGHT_METHODS = [UNITARY_WEIGHTS, CO_OCCURRENCES_WEIGHTS, CANDIDATE_SIMILARITY_WEIGHTS]

NONE_DOCUMENT_EXCEPTION = "Document has None value"
NONE_CANDIDATES_EXCEPTION = "List of candidates has None value"
ZERO_CANDIDATES_EXCEPTION = "Empty list of candidates"
WEIGHT_METHOD_EXCEPTION = "Invalid weight method"
BACKGROUND_COLL_REQUIRED = "You need to specify background collection using that weight method"

WORDS_VECTORS_PATH = "files/wiki-news-300d-1M.vec"
WORDS_VECTOR_PICKLE = "files/words_vectors.p"


def load_vectors(fname, n_words):
    with open(fname, "r", encoding="UTF-8") as file:
        # file first line
        total_words, vectors_dim = file.readline().split()
        print("total number of words: ", total_words)
        print("vectors dimensionality: ", vectors_dim)

        # get first n_words vectors
        data_dict = {}

        while len(data_dict) < n_words:
            print("reading word ", len(data_dict))
            line_values = file.readline().split()
            word = line_values[0]
            vector = np.array(line_values[1:]).astype(np.float)
            data_dict[word] = vector

        return data_dict


class GraphBuilder:

    def __init__(self, document: str, background_collection: list, candidates: np.ndarray, weight_method: int,
                 n: int) -> None:
        # validate parameters
        if document is None:
            raise ValueError(NONE_DOCUMENT_EXCEPTION)

        elif candidates is None:
            raise ValueError(NONE_CANDIDATES_EXCEPTION)

        elif len(candidates) == 0:
            raise ValueError(ZERO_CANDIDATES_EXCEPTION)

        elif weight_method not in POSSIBLE_WEIGHT_METHODS:
            raise ValueError(WEIGHT_METHOD_EXCEPTION)

        elif (background_collection is None or len(background_collection)) == 0 \
                and weight_method == CO_OCCURRENCES_WEIGHTS:
            raise ValueError(BACKGROUND_COLL_REQUIRED)

        # initialize attributes
        self.__document = document
        self.__background_collection = background_collection
        self.__candidates = np.array(candidates)
        self.__n = n
        self.__weight_matrix = self.__calculate_weight_matrix(weight_method)
        self.__current_sentence_nodes = []
        self.__graph = nx.Graph()
        self.__node_sentence_indexes = {}

    @staticmethod
    def split_into_sentences(document: str) -> list:
        return nltk.sent_tokenize(document)

    def build_graph(self) -> nx.Graph:
        doc_sentences = GraphBuilder.split_into_sentences(self.__document)

        for sentence_i in range(len(doc_sentences)):
            sentence = doc_sentences[sentence_i]
            self.__current_sentence_nodes = []

            try:
                sentence_tokens = TfidfVectorizer(stop_words="english", ngram_range=(1, self.__n)).fit([sentence]) \
                    .get_feature_names()

            except ValueError:  # sentence with only stop words
                continue

            for candidate in self.__candidates:
                # candidate found in current sentence
                if candidate in sentence_tokens:
                    self.__add_node(candidate)
                    self.__update_node_edges(candidate)
                    self.__current_sentence_nodes.append(candidate)

                    # if candidate was not seen in previous sentences
                    if candidate not in self.__node_sentence_indexes.keys():
                        self.__node_sentence_indexes[candidate] = len(doc_sentences) - sentence_i

        return self.__graph

    def reset_obj(self, document: str, background_collection: list, candidates: np.ndarray, weight_method: int,
                  n: int) -> None:
        self.__init__(document, background_collection, candidates, weight_method, n)

    def get_candidates_sentences(self):
        return self.__node_sentence_indexes

    def __add_node(self, candidate: str) -> None:
        if candidate not in self.__graph.nodes():
            self.__graph.add_node(candidate)

    def __update_node_edges(self, candidate: str) -> None:
        if len(self.__current_sentence_nodes) > 0:
            node_i = np.where(self.__candidates == candidate)[0][0]

            for each_node in self.__current_sentence_nodes:
                if (candidate, each_node) not in self.__graph.edges() and \
                        (each_node, candidate) not in self.__graph.edges():
                    each_node_i = np.where(self.__candidates == each_node)
                    edge_weight = self.__weight_matrix[node_i, each_node_i]
                    edge = [(candidate, each_node, edge_weight)]
                    self.__graph.add_weighted_edges_from(edge)

    def __calculate_weight_matrix(self, weight_method):
        nr_candidates = len(self.__candidates)

        if weight_method == UNITARY_WEIGHTS:
            return np.ones((nr_candidates, nr_candidates), dtype=float)

        elif weight_method == CO_OCCURRENCES_WEIGHTS:
            counter_matrix = np.zeros((nr_candidates, nr_candidates), dtype=int)

            for doc in self.__background_collection:
                doc_tokens = TfidfVectorizer(stop_words="english", ngram_range=(1, self.__n)).fit([doc]) \
                    .get_feature_names()

                for c1_i in range(nr_candidates):
                    for c2_i in range(nr_candidates):
                        candidate1 = self.__candidates[c1_i]
                        candidate2 = self.__candidates[c2_i]

                        if candidate1 in doc_tokens and candidate2 in doc_tokens:
                            counter_matrix[c1_i, c2_i] += 1

            return counter_matrix

        elif weight_method == CANDIDATE_SIMILARITY_WEIGHTS:
            sim_matrix = np.zeros((nr_candidates, nr_candidates), dtype=float)
            vectors: dict

            with open("../" + WORDS_VECTOR_PICKLE, "rb") as file:
                vectors = pickle.load(file)

            possible_words = vectors.keys()

            for c1_i in range(nr_candidates):
                for c2_i in range(nr_candidates):
                    candidate1 = self.__candidates[c1_i]
                    candidate2 = self.__candidates[c2_i]

                    if candidate1 in possible_words and candidate2 in possible_words:
                        c1_vector = vectors[self.__candidates[c1_i]]
                        c2_vector = vectors[self.__candidates[c2_i]]
                        cosine_sim = -cosine(c1_vector, c2_vector) + 1
                        sim_matrix[c1_i, c2_i] = cosine_sim

            return sim_matrix


if __name__ == "__main__":
    n = 300000
    words_vectors = load_vectors(WORDS_VECTORS_PATH, n)

    with open(WORDS_VECTOR_PICKLE, "wb") as file:
        pickle.dump(words_vectors, file, protocol=pickle.HIGHEST_PROTOCOL)
