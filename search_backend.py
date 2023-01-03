import pickle
import inverted_index_gcp

with open('anchor_index.pkl', 'rb') as f:
    anchor_index = pickle.load(f)

with open('body_index.pkl', 'rb') as f:
    body_index = pickle.load(f)

with open('title_index.pkl', 'rb') as f:
    title_index = pickle.load(f)

print(type(anchor_index))

