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
nltk.download('stopwords')

#TODO: Remove this later
import requests
from time import time

class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)



app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

# Initializations

# Reading the indexes
# global body_index
# global anchor_index
# global title_index
body_index = InvertedIndex.read_index('.', "body_index")
anchor_index = InvertedIndex.read_index('.', "anchor_index")
title_index = InvertedIndex.read_index('.', "title_index")

# Paths to .bin files
# path_anchor = "/content/postings_gcp_anchor"
# path_body = "/content/postings_gcp_body"
# path_title = "/content/postings_gcp_title"
path_anchor = "postings_gcp_anchor"
path_body = "postings_gcp_body"
path_title = "postings_gcp_title"

body_index.bin_path = path_body
anchor_index.bin_path = path_anchor
title_index.bin_path = path_title


# Loading the dictionaries
with open('doc_id_title_dict.pickle', 'rb') as f:
    # global doc_title_dict
    doc_title_dict = pickle.load(f)

with open('doc_len_dict.pickle', 'rb') as f:
    # global doc_len_dict
    doc_len_dict = pickle.load(f)

with open('page_views_dict.pkl', 'rb') as f:
    # global page_views_dict
    page_views_dict = pickle.load(f)

with open('page_rank_dict.pickle', 'rb') as f:
    # global page_rank_dict
    page_rank_dict = pickle.load(f)

with open('doc_norm_dict.pickle', 'rb') as f:
    # global doc_norm_dict
    doc_norm_dict = pickle.load(f)

with open('query_expansion_dict.pickle', 'rb') as f:
    # global query_expansion_dict
    query_expansion_dict = pickle.load(f)

print("All files loaded successfully, Ready to go!")

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
    weights = [0.46162036, 1.42513415, 1.13945286, 1.87296152, 1.39693174]
    BODY_WEIGHT, ANCHOR_WEIGHT, TITLE_WEIGHT, PAGE_RANK_WEIGHT, PAGE_VIEW_WEIGHT = weights

    if query[-1] == '?':
        BODY_WEIGHT *= 2

    # Tokenizing the query
    tokens = tokenize(query)

    tokens_in_doc_title = collections.Counter()
    tokens_in_doc_anchor = collections.Counter()

    for token in tokens:

        # Searching the term in anchor index
        if anchor_index.df.get(token, None):
            for doc_tf in anchor_index.read_posting_list(token):
                tokens_in_doc_anchor[doc_tf[0]] += 1
                similarity_dict[doc_tf[0]] = similarity_dict.get(doc_tf[0], 0) + math.pow(ANCHOR_WEIGHT,tokens_in_doc_anchor[doc_tf[0]])

        # Searching the term in title index
        if title_index.df.get(token, None):
            for doc_tf in title_index.read_posting_list(token):
                tokens_in_doc_title[doc_tf[0]] += 1
                similarity_dict[doc_tf[0]] = similarity_dict.get(doc_tf[0], 0) + math.pow(TITLE_WEIGHT,tokens_in_doc_title[doc_tf[0]])

        # Searching the term in body index
        if body_index.df.get(token, None):
            for doc_tf in body_index.read_posting_list(token):
                c = bm25_update(token, doc_tf[0], doc_tf[1]) * BODY_WEIGHT
                similarity_dict[doc_tf[0]] = similarity_dict.get(doc_tf[0], 0) + c

        # Adding page_rank and page_views of each relevant document
        for doc_id in similarity_dict.keys():
            similarity_dict[doc_id] += page_rank_dict.get(doc_id, 0) / page_rank_max * PAGE_RANK_WEIGHT
            similarity_dict[doc_id] += page_views_dict.get(doc_id, 0) / page_views_max * PAGE_VIEW_WEIGHT

    top_n_results(similarity_dict, res, 50)

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
        if body_index.df.get(token, None):
            for doc_tf in body_index.read_posting_list(token):
                if doc_tf[0] not in similarity_dict:
                    similarity_dict[doc_tf[0]] = 0
                similarity_dict[doc_tf[0]] += tf_idf_calc(token, doc_tf[0], doc_tf[1])

    for doc in similarity_dict.keys():
        similarity_dict[doc] = similarity_dict[doc] * normalize(tokens) * doc_norm_dict[doc]

    top_n_results(similarity_dict, res, 50)

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
        if title_index.df.get(token, None):
            for doc_id_tf in title_index.read_posting_list(token):
                doc_id = doc_id_tf[0]
                title = doc_title_dict[doc_id]
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
        if anchor_index.df.get(token, None):
            for doc_id_tf in anchor_index.read_posting_list(token):
                doc_id = doc_id_tf[0]
                if not doc_title_dict.get(doc_id,None):
                    continue
                title = doc_title_dict[doc_id]
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

    res = [page_rank_dict.get(doc_id, -1) for doc_id in wiki_ids]
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

    res = [page_views_dict.get(doc_id, -1) for doc_id in wiki_ids]
    res = list(filter(lambda x: x != -1, res))

    # END SOLUTION
    return jsonify(res)

# Global variables
english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

N = 6348910
page_rank_max = max(page_rank_dict.values())
page_views_max = max(page_views_dict.values())


# Helping functions
def top_n_results(similarity_dict, res, n=100):
    # Top up to 100
    res_size = min(n, len(similarity_dict.keys()))
    top100 = list(sorted(similarity_dict.items(), key=lambda item: item[1], reverse=True)[:res_size])
    # Returning the top up to 100 tuples -> (ID, Tittle)
    for pair in top100:
        res.append((pair[0], doc_title_dict[pair[0]]))


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
    tf = tf / doc_len_dict[doc_id]
    #     idf = math.log(N/body_index.df[token], 2)
    idf = math.log(N / body_index.df[token], 2)
    return tf * idf


def bm25_update(token, doc_id, tf):
    k1 = 1.2
    B = 0.75
    N = 6348910
    avgl = 319.5242353411845
    idf = math.log(((N - body_index.df[token] + 0.5) / (body_index.df[token] + 0.5)) + 1, math.e)
    score = (idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - B + B * (doc_len_dict[doc_id] / avgl)))))
    return score


def query_expansion(tokenized_query, k):
    if k <= 0:
        return tokenized_query
    # tokenized_query = tokenize(query)
    expansion_list = []
    for token in tokenized_query:
        token_expansion = query_expansion_dict.get(token, None)
        tmp = k
        if not token_expansion:
            continue
        for i in range(len(token_expansion)):
            if tmp <= 0:
                break
            expansion_list.append(token_expansion[i][1])
            tmp -= 1

    tokenized_query.update(expansion_list)
    return tokenized_query
#
# Query Expansion with word2vec trial
# tokenized_query = tokenize(query)
# q_len = len(tokenized_query)
# expand_factor = 0
# if q_len == 0:
#     return jsonify(res)
# if q_len == 1:
#     expand_factor = 2
# elif q_len == 2 or q_len == 3:
#     expand_factor = 1
# else:
#     expand_factor = 0
# tokens = query_expansion(tokenized_query, expand_factor)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
