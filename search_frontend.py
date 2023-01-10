import pickle
from google.cloud import storage
from flask import Flask, request, jsonify
import nltk
from nltk.stem.po   rter import *
from nltk.corpus import stopwords
import re
from collections import Counter
import math


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        bucket_name = "ir_assg3_eithan"
        client = storage.Client()
        self.bucket = client.bucket(bucket_name=bucket_name)
        for blob in client.list_blobs(bucket_name, prefix='indexes/'):
            if 'body_index' in blob.name:
                with blob.open('rb') as f:
                    self.body_index = pickle.load(f)
            if 'anchor_index' in blob.name:
                with blob.open('rb') as f:
                    self.anchor_index = pickle.load(f)
            if 'title_index' in blob.name:
                with blob.open('rb') as f:
                    self.title_index = pickle.load(f)
            if 'doc_len_dict' in blob.name:
                with blob.open('rb') as f:
                    self.doc_len_dict = pickle.load(f)
            # if 'page_rank_dict' in blob.name:
            #     with blob.open('rb') as f:
            #         self.page_rank_dict = pickle.load(f)
            if 'page_views_dict' in blob.name:
                with blob.open('rb') as f:
                    self.page_views_dict = pickle.load(f)
            if 'doc_id_title_dict' in blob.name:
                with blob.open('rb') as f:
                    self.doc_id_title_dict = pickle.load(f)
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

# CHANGEEEE
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

    tokens = tokenize(query)
    # word2vec
    # Constants
    BODY_WEIGHT = 0.3
    ANCHOR_WEIGHT = 0.35
    TITLE_WEIGHT = 0.35
    token_doc_titles_occurrences = Counter()
    token_doc_anchor_occurrences = Counter()

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

            a = token_doc_anchor_occurrences.get(doc_tf[0], 0) * ANCHOR_WEIGHT
            b = token_doc_titles_occurrences.get(doc_tf[0], 0) * TITLE_WEIGHT
            c = bm25_update(token, doc_tf[0], doc_tf[1]) * BODY_WEIGHT
            update =  a + b + c
            similarity_dict[doc_tf[0]] += update


    for doc in similarity_dict.keys():
        similarity_dict[doc] = similarity_dict[doc] * normalize(tokens) * app.body_index.doc_norm_dict[doc]

    # END SOLUTION
    return jsonify(res)


def bm25_update(token, doc_id, tf):
    k1 = 1.5
    B = 0.75
    N = 6348910
    avgl = 319.5242353411845
    idf = math.log(((N - app.body_index.df[token] + 0.5)/(app.body_index.df[token]+0.5)) + 1, math.e)
    score = (idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - B + B * (DL[doc_id] / avgl)))))
    return score


def normalize(tokens):
    counter = Counter()
    for token in tokens:
        counter[token] += 1
    norm = 0
    for value in counter.values():
        norm += value * value
    return 1 / (math.sqrt(norm))


def tf_idf_calc(token, doc_id, tf):
    tf = tf / doc_len[doc_id]
    idf = math.log(6348910 / app.body_index.df[token], 2)
    return tf * idf

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
        similarity_dict[doc] = similarity_dict[doc]*normalize(tokens)*app.body_index.doc_norm_dict[doc]

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

    counter = {}
    for token in tokenize(query):
        for tup in app.title_index.read_posting_list(token):
            if tup not in counter:
                counter[tup] = 0
            counter[tup] += 1

    res = (sorted(counter.values(), reverse=True))

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

    counter = {}
    for token in tokenize(query):
        for tup in app.anchor_index_index.read_posting_list(token):
            if tup not in counter:
                counter[tup] = 0
            counter[tup] += 1

    res = (sorted(counter.values(), reverse=True))

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

    res = [app.page_rank_dict.get(doc_id, -1) for doc_id in wiki_ids]
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

    # END SOLUTION
    return jsonify(res)

def tokenize_binary(text):
    tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
    tokens = set(tok for tok in tokens if tok not in all_stopwords)
    return tokens

def tokenize(text):
    tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
    return tokens

if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)





