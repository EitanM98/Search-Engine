# About
This report describes the development and optimization of a search engine as part of an Information Retrieval course at Ben Gurion University.
The project was submitted by Ariel Dawidowicz and Eithan Markman. The project is focused on implementing a search engine using information retrieval techniques.
The search engine retrieves information from the entire English Wikipedia corpus, includes all Wiki pages on August 2021.


# Workflow
The project went through a process of several iterations and improvements, starting with a basic approach and then moving on to more advanced techniques. Here is a summary of the different versions:

## Inverted Indexes
- Body index
- Anchor index
- Title index

## Dictionaries
- doc_lec
- query_expansion
- doc_title
- doc_norm

V1: Implemented search based on all the indexes (body, anchor, title) using BM25 similarity function and page rank and page views as additional features.
However, the results of the MAP@40 ranking were not satisfactory.

V2: Normalization of page rank and page views was changed, and weight of different features was adjusted.
This led to better performance in precision.

V3: Query expansion method was used using Fasttext word2vec model. However, the results were less good than the previous try.

V4: Query expansion was used only for search on the body. The results were still bad.

V5: Added a case that, if the query ends with “?” more importance was given to the body and different approach on the TITLE_WEIGHT and ANCHOR_WEIGHT was taken.

V6: Used SciPy library’s minimize function to optimize the weight and get better results.

# Process graph
![image](https://user-images.githubusercontent.com/101277239/212707561-d0e73527-8f81-4844-974c-7c14088a2065.png)


# Technologies
The following technologies were used in this project:

- Python: As the programming language
- Pyspark: for large-scale data processing
- Fasttext word2vec model: for query expansion
- GCP bucket: for storing the data
- Flask: for creating web application
- SciPy library's minimize function: for optimization of the weight
- Github: as the version control system.

# Conclusion
This project demonstrates the implementation of a search engine using information retrieval techniques.
The team went through several iterations and improvements to the initial implementation, using various techniques such as normalization,
weight adjustments, query expansion and optimization. The final version of the engine shows improved performance in terms of precision.
However, the team faced some limitations and challenges, as some techniques did not provide the expected results.
The project can be used as a basis for further research and development in the field of information retrieval.


