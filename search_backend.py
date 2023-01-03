import pickle
import inverted_index_gcp

with open('small_index/indexes/anchor_index.pkl', 'rb') as f:
    anchor_index = pickle.load(f)

with open('small_index/indexes/body_index.pkl', 'rb') as f:
    body_index = pickle.load(f)

with open('small_index/indexes/title_index.pkl', 'rb') as f:
    title_index = pickle.load(f)

# print(type(anchor_index))
print(anchor_index.posting_locs)

