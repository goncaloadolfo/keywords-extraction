"""
Web server serving html page with NYT articles key phrases.
"""

from urllib.request import urlopen
from xml.dom import minidom

import tornado.ioloop
from networkx import pagerank
from sklearn.feature_extraction.text import TfidfVectorizer
from tornado.web import RequestHandler, Application

from graph_constructor import UNITARY_WEIGHTS, GraphBuilder
from utils import PRIORS_UNIFORM, get_keyphrases

# url and global cache variables
SERVER_PORT = 8888
RSS_URL = "https://rss.nytimes.com/services/xml/rss/nyt/Sports.xml"
cached_articles = None
cached_page = None

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
        global cached_articles, cached_page

        try:
            rss = request_rss()
            articles = read_news(rss)

            if check_articles_change(articles):
                keyphrases = extract_keyphrases(articles)
                html_page = build_html_page(articles, keyphrases)
                self.write(html_page)

                cached_articles = articles
                cached_page = html_page

            else:
                self.write(cached_page)

        except Exception:
            self.send_error()


def check_articles_change(articles: list) -> bool:
    return True


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


def build_html_page(articles: list, keyphrases: list) -> str:
    html_page = ""

    for article_i in range(len(articles)):
        article = articles[article_i]
        article_keyphrases = keyphrases[article_i]

        title = article.get_title()
        link = article.get_link()

        html_article = "<div> " \
                       "<p>{}</p>" \
                       "<p>{}</p>" \
                       "<a href='{}'>article link</a>" \
                       "</div>".format(title, article_keyphrases, link)
        html_page += html_article

    return html_page


def simple_test():
    rss = request_rss()
    print("rss: ", rss)

    articles = read_news(rss)
    print("number of articles: ", len(articles))
    print("example article: title={}, description={}".format(articles[0].get_title(), articles[0].get_description()))

    articles_keyphrases = extract_keyphrases(articles)
    print("keyphrases for example doc: ", articles_keyphrases[0])

    html_page = build_html_page(articles, articles_keyphrases)
    print("html page: ", html_page)


def main():
    # simple_test()

    app = Application([(r"/", PageHandler)])
    app.listen(SERVER_PORT)
    print("server listening on port {} ...".format(SERVER_PORT))
    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    main()
