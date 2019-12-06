"""
Exercise 1 - keyphrase extraction based in graph ranking.
"""

import matplotlib.pyplot as plt
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer

from doc_reader import read_doc_from_disk, TEST_DOCUMENT_PATH
from graph_constructor import GraphBuilder, UNITARY_WEIGHTS
from utils import get_keyphrases

graph_path = "../../results/ex1_graph.png"
d = 0.15
n = 1
max_iter = 50

document = read_doc_from_disk("../" + TEST_DOCUMENT_PATH)
candidates = TfidfVectorizer(stop_words="english", ngram_range=(1, n)).fit([document]).get_feature_names()

graph_builder_obj = GraphBuilder(document, None, candidates, UNITARY_WEIGHTS, n, None)
doc_graph = graph_builder_obj.build_graph()

final_prestige = nx.pagerank(doc_graph, alpha=1.0-d, max_iter=max_iter, weight='weight')
keyphrases = get_keyphrases([final_prestige])

nx.draw(doc_graph)
plt.savefig(graph_path)

print("Document: ", document)
print()
print("Candidates: ", candidates)
print()
print("Graph nodes: ", doc_graph.nodes())
print()
print("Graph edges: ", doc_graph.edges())
print()
print("Page rank output: ", final_prestige)
print()
print("Top 5 keyphrases: ", keyphrases)
