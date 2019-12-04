"""
Web server serving html page with NYT articles key phrases.
"""

import operator
from urllib.request import urlopen
from xml.dom import minidom

import numpy as np
import tornado.ioloop
from networkx import pagerank
from sklearn.feature_extraction.text import TfidfVectorizer
from tornado.web import RequestHandler, Application, StaticFileHandler

from graph_constructor import UNITARY_WEIGHTS, GraphBuilder
from utils import PRIORS_UNIFORM, get_keyphrases

# server port, rss url and web files paths
SERVER_PORT = 8888
RSS_URL = "https://rss.nytimes.com/services/xml/rss/nyt/Sports.xml"
TEMPLATE_PATH = "./tornado_templates/index.html"
STYLESHEET_FOLDER = "./css"
JS_FOLDER = "./js"

# xml tags
ARTICLE_TAG = "item"
TITLE_TAG = "title"
DESCR_TAG = "description"
ARTICLE_LINK_TAG = "link"

# algorithm options
n = 1  # 1 <= n <= 3
d = 0.15
max_iter = 50
weight_method = UNITARY_WEIGHTS  # {UNITARY_WEIGHTS, CO_OCCURRENCES_WEIGHTS, CANDIDATE_SIMILARITY_WEIGHTS}
priors_method = PRIORS_UNIFORM  # {PRIORS_UNIFORM, PRIORS_SENTENCE_POS, PRIORS_TFIDF}


class PageHandler(RequestHandler):

    def get(self):
        rss = request_rss()
        articles = read_news(rss)
        keyphrases = extract_keyphrases(articles)
        ocurrences = count_ocurrences(keyphrases)
        dict_encoded_articles = dict_encode(articles, keyphrases)
        self.render(TEMPLATE_PATH, articles=articles, keyphrases=keyphrases, kp_ocurrences=ocurrences,
                    dict_encoded_articles=dict_encoded_articles)


def request_rss() -> str:
    conn = urlopen(RSS_URL)
    return str(conn.read(), 'utf-8')


def read_news(xml_rss: str) -> list:
    class Article:
        def __init__(self, title, description, link):
            self.__title = title
            self.__description = description
            self.__link = link

        def get_title(self):
            return self.__title

        def get_description(self):
            return self.__description

        def get_link(self):
            return self.__link

    xml_file = minidom.parseString(xml_rss)
    articles = xml_file.getElementsByTagName(ARTICLE_TAG)

    article_objs = []
    for article in articles:
        title_str = article.getElementsByTagName(TITLE_TAG)[0].firstChild.data
        description_str = article.getElementsByTagName(DESCR_TAG)[0].firstChild.data
        link = article.getElementsByTagName(ARTICLE_LINK_TAG)[0].firstChild.data
        article_objs.append(Article(title_str, description_str, link))

    return article_objs


def extract_keyphrases(article_objs: list) -> list:
    results = []

    for article_i in range(len(article_objs)):
        article = article_objs[article_i]
        title = article.get_title()
        description = article.get_description()

        document = title + description if title[-1] == '.' else title + "." + description
        candidates = TfidfVectorizer(stop_words="english", ngram_range=(1, n)).fit([document]).get_feature_names()
        doc_graph = GraphBuilder(document, None, candidates, weight_method, n).build_graph()
        final_prestige = pagerank(doc_graph, alpha=1.0 - d, max_iter=max_iter, weight='weight')
        keyphrases = get_keyphrases([final_prestige])
        results.append(keyphrases)

    return results


def dict_encode(articles: list, keyphrases: list) -> list:
    d = []

    for i in range(len(articles)):
        article = articles[i]

        title = article.get_title()
        link = article.get_link()
        kps = keyphrases[i]

        article_dict = {'title': title, 'link': link, 'kps': kps}
        d.append(article_dict)

    return d


def count_ocurrences(articles_keyphrases: list) -> list:
    unique_keyphrases = np.unique(articles_keyphrases)
    kps_counter = []

    for keyphrase_i in range(len(unique_keyphrases)):
        keyphrase = unique_keyphrases[keyphrase_i]
        counter = 0

        for article_kps in articles_keyphrases:
            if keyphrase in article_kps:
                counter += 1

        kps_counter.append((keyphrase, counter))

    return sorted(kps_counter, key=operator.itemgetter(1), reverse=True)


def simple_test():
    rss = request_rss()
    print("rss: ", rss)

    articles = read_news(rss)
    print("number of articles: ", len(articles))
    print("example article: title={}, description={}".format(articles[0].get_title(), articles[0].get_description()))

    articles_keyphrases = extract_keyphrases(articles)
    print("keyphrases for example doc: ", articles_keyphrases[0])


def main():
    # simple_test()

    handlers = [(r"/", PageHandler),
                ("/css/(.*)", StaticFileHandler, {"path": STYLESHEET_FOLDER},),
                ("/js/(.*)", StaticFileHandler, {"path": JS_FOLDER},)]

    app = Application(handlers)
    app.listen(SERVER_PORT)
    print("server listening on port {} ...".format(SERVER_PORT))
    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    main()
