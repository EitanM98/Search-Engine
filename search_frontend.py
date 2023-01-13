import pickle
from google.cloud import storage
from flask import Flask, request, jsonify
import nltk
from nltk.corpus import stopwords
import re
from collections import Counter
import math
import collections
import json
from inverted_index_gcp import *


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        # Initializations

        # Reading the indexes
        self.body_index = InvertedIndex.read_index('.', "body_index")
        self.anchor_index = InvertedIndex.read_index('.', "anchor_index")
        self.title_index = InvertedIndex.read_index('.', "title_index")

        # Paths to .bin files
        path_anchor = "/content/postings_gcp_anchor"
        path_body = "/content/postings_gcp_body"
        path_title = "/content/postings_gcp_title"

        self.body_index.bin_path = path_body
        self.anchor_index.bin_path = path_anchor
        self.title_index.bin_path = path_title

        # Loading the dictionaries
        with open('doc_id_title_dict.pickle', 'rb') as f:
            self.doc_title_dict = pickle.load(f)

        with open('doc_len_dict.pickle', 'rb') as f:
            self.doc_len_dict = pickle.load(f)

        with open('page_views_dict.pkl', 'rb') as f:
            self.page_views_dict = pickle.load(f)

        with open('page_rank_dict.pickle', 'rb') as f:
            self.page_rank_dict = pickle.load(f)

        with open('doc_norm_dict.pickle', 'rb') as f:
            self.doc_norm_dict = pickle.load(f)

        with open('query_expansion_dict.pickle', 'rt') as f:
            self.query_expansion_dict = pickle.load(f)

        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

# Global variables
english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

N = 6348910
page_rank_max = max(app.page_rank_dict.values())
page_views_max = max(app.page_views_dict.values())


@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)

    # BEGIN SOLUTION

    similarity_dict = {}

    # Constants
    weights = [0.55754871, 1.25785412, 0.54434859, 0.09795087, 0.38754463]
    BODY_WEIGHT, ANCHOR_WEIGHT, TITLE_WEIGHT, PAGE_RANK_WEIGHT, PAGE_VIEW_WEIGHT = weights

    token_doc_titles_occurrences = Counter()
    token_doc_anchor_occurrences = Counter()

    # Tokenizing the query
    tokens = tokenize(query)

    # TODO:Query expansion
    for token in tokens:
        # Searching the term in anchor index
        for doc_tf in app.anchor_index.read_posting_list(token):
            token_doc_anchor_occurrences[doc_tf[0]] = 1 + token_doc_anchor_occurrences.get(doc_tf[0], 0)

        # Searching the term in title index
        for doc_tf in app.title_index.read_posting_list(token):
            token_doc_titles_occurrences[doc_tf[0]] = 1 + token_doc_titles_occurrences.get(doc_tf[0], 0)

        # Searching the term in body index
        for doc_tf in app.body_index.read_posting_list(token):
            if doc_tf[0] not in similarity_dict.keys():
                similarity_dict[doc_tf[0]] = 0

            # Calculating the summation of the similarities
            a = token_doc_anchor_occurrences.get(doc_tf[0], 0) * ANCHOR_WEIGHT
            b = token_doc_titles_occurrences.get(doc_tf[0], 0) * TITLE_WEIGHT
            c = bm25_update(token, doc_tf[0], doc_tf[1]) * BODY_WEIGHT
            d = app.page_rank_dict.get(doc_tf[0], 0) / page_rank_max * PAGE_RANK_WEIGHT
            e = app.page_views_dict.get(doc_tf[0], 0) / page_views_max * PAGE_VIEW_WEIGHT

            update = a + b + c + d + e
            similarity_dict[doc_tf[0]] += update

    top_n_results(similarity_dict, res, 100)

    # END SOLUTION
    return jsonify(res)


@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. 

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    similarity_dict = {}
    tokens = tokenize(query)
    for token in tokens:
        for doc_tf in app.body_index.read_posting_list(token):
            if doc_tf[0] not in similarity_dict:
                similarity_dict[doc_tf[0]] = 0
            similarity_dict[doc_tf[0]] += tf_idf_calc(token, doc_tf[0], doc_tf[1])

    for doc in similarity_dict.keys():
        similarity_dict[doc] = similarity_dict[doc] * normalize(tokens) * app.body_index.doc_norm_dict[doc]

    top_n_results(similarity_dict, res, 100)

    # END SOLUTION
    return jsonify(res)



@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        QUERY WORDS that appear in the title. For example, a document with a 
        title that matches two of the query words will be ranked before a 
        document with a title that matches only one query term. 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''

    # query=["hello world"] - > [(wiki_id, title)] sorted decreasing
    # in a way that the title that contains most words of the query

    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    counter = collections.Counter()
    for token in tokenize(query):
        for doc_id_tf in app.title_index.read_posting_list(token):
            doc_id = doc_id_tf[0]
            title = app.doc_title_dict[doc_id]
            counter[(doc_id, title)] += 1

    res = list(map(lambda tup: tup[0], counter.most_common()))

    # END SOLUTION
    return jsonify(res)


@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        For example, a document with a anchor text that matches two of the 
        query words will be ranked before a document with anchor text that 
        matches only one query term. 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''

    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    counter = collections.Counter()
    for token in tokenize(query):
        for doc_id_tf in app.anchor_index.read_posting_list(token):
            doc_id = doc_id_tf[0]
            title = app.doc_title_dict[doc_id]
            counter[(doc_id, title)] += 1

    res = list(map(lambda tup: tup[0], counter.most_common()))

    # END SOLUTION
    return jsonify(res)


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    # TODO: check the forum for the answer
    res = [app.page_rank_dict.get(doc_id, -1) for doc_id in wiki_ids]
    res = list(filter(lambda x: x != -1, res))
    # END SOLUTION
    return jsonify(res)


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''

    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    res = [app.page_views_dict.get(doc_id, -1) for doc_id in wiki_ids]
    res = list(filter(lambda x: x != -1, res))

    # END SOLUTION
    return jsonify(res)


# Helping functions
def top_n_results(similarity_dict, res, n=100):
    # Top up to 100
    res_size = min(n, len(similarity_dict.keys()))
    top100 = list(sorted(similarity_dict.items(), key=lambda item: item[1], reverse=True)[:res_size])
    # Returning the top up to 100 tuples -> (ID, Tittle)
    for pair in top100:
        res.append((pair[0], app.doc_title_dict[pair[0]]))


def tokenize_binary(text):
    tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
    tokens = set(tok for tok in tokens if tok not in all_stopwords)
    return tokens


def tokenize(text):
    tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
    tokens = set(tok for tok in tokens if tok not in all_stopwords)
    # tokens = list(tok for tok in tokens if tok not in all_stopwords) TODO: check for list or set
    return tokens


def normalize(tokens):
    counter = Counter(tokens)
    norm = 0
    for value in counter.values():
        norm += value * value
    if norm == 0:
        return 0
    return 1 / (math.sqrt(norm))


def tf_idf_calc(token, doc_id, tf):
    tf = tf / app.doc_len_dict[doc_id]
    #     idf = math.log(N/app.body_index.df[token], 2)
    idf = math.log(N / app.body_index.df[token], 2)
    return tf * idf


def bm25_update(token, doc_id, tf):
    k1 = 1.5
    B = 0.75
    N = 6348910
    avgl = 319.5242353411845
    idf = math.log(((N - app.body_index.df[token] + 0.5) / (app.body_index.df[token] + 0.5)) + 1, math.e)
    score = (idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - B + B * (app.doc_len_dict[doc_id] / avgl)))))
    return score


def query_expansion(query):


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
