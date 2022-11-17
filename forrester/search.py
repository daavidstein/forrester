from typing import List
from flask import Flask,request
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from numpy import matrix
import logging
from werkzeug.routing import IntegerConverter # noqa - needed to enforce integer types for ids

def create_similarity_mapping(articles:pd.DataFrame, top_n: int, outpath: str="sorted_indices.npy") -> np.array:
    """TODO cache and run once per day
           database?
           precompute mapping for top 100?
    """
    vectors = np.array((articles["vector"].tolist()))
    """for each similarity vector `i` and for each entry `j`,
    the `j`th entry gives the similarity (cosine of the angle) between the vector 
    at index i in `vectors` and the vector at index j in `vectors`. 
    Hence for a given title we want to return the indices `j` which are largest (most similar)"""
    similarity_matrix = cosine_similarity(vectors)
    #Set the diagonal to 0 so that we exclude the ith article from the search results
    #the most similar article to X is itself, but that's a useless result.
    np.fill_diagonal(similarity_matrix,0)
    #get  an an array of where the ith entry corresponds to the indices of the top N similar articles (excluding self)
    #in the original df of vectors
    top_indices = matrix.argpartition(similarity_matrix, -top_n,axis=0)[-top_n:].transpose()
    # these are the similarity values which correspond to the top indices
    # now we want to sort all these and use them to sort our indices
    top_values = np.take_along_axis(similarity_matrix, top_indices, 1)
    #negate the top n so that we get the indices in descending order
    sorted_indices = np.take_along_axis(top_indices,matrix.argsort(-top_values),axis=1)
    np.save(outpath,sorted_indices)
    logger.info(f"similarity mapping saved to {outpath}")
    return sorted_indices


def vector_to_id(vector: List[int],articles: pd.DataFrame):
    """Look up the id corresponding to a vector"""
    #need to convert lists to tuples to compare with pandas
    mask = articles.vector.map(tuple) == tuple(vector)
    rows_matched_to_vector = articles[mask].index
    assert len(rows_matched_to_vector) == 1, f"did not find a unique id corresponding to vector!\nVector: {vector}"
    id_ = rows_matched_to_vector[0]
    return id_


"""begin main app"""
TOP_N = 5
PATH_TO_ARTICLES = "Forrester_mle2_dataset.json"
logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)
app = Flask(__name__)

#TODO cache and refresh once per day
# "reactive"?
articles = pd.read_json(PATH_TO_ARTICLES, lines=False)
articles = articles.set_index("id")
similarity_mapping = create_similarity_mapping(articles,TOP_N)

def test_create_similarity_mapping():
    #TODO
    pass

@app.route('/search/<int:id_>', methods=['GET'])
@app.route('/search/<int:id_>/', methods=['GET'])
def search_by_id_default(id_:int):
    # we allow POST method since this is called by search_by_vector
    if request.method in ['GET', 'POST']:
         return search_by_id(id_,TOP_N)


@app.route('/search/<int:id_>/<int:num_matches>', methods=['GET'])
@app.route('/search/<int:id_>/<int:num_matches>/', methods=['GET'])
def search_by_id(id_:int, num_matches:int):
    # we allow POST method since this is called by search_by_vector (via search_by_id_default)
    if request.method in ['GET', 'POST']:
        similarity_mapping  = create_similarity_mapping(articles,num_matches)
        try:
            row_num_for_id = articles.index.get_loc(id_)
            top_matches= articles.iloc[similarity_mapping[row_num_for_id]][["title"]]
            response = top_matches.reset_index().to_dict(orient='records')
        except KeyError as e:
            response = {"KeyError": f"Unable to find an article corresponding to id {id_}"}
        except Exception as e:
            response = {"error":str(e)}

        return response


@app.route("/")
def hello_world():
    #FIXME would be good to have this display api docs
    # or at the very least a readme
    return "<p>Hello, World!</p>"

@app.route('/search', methods=['POST'])
@app.route('/search/', methods=['POST'])
def search_by_vector():
    """
    Get the top 5 articles whose semantic content vector are most similar to the provided vector

    Note: I've included this function because the spec called for it - however, I think it makes more sense to  search by id - this allows
    a 'GET' request, which more accurately describes the action of searching - (we want to retrieve something) and facilitates
    including the top_n argument in the request. I've therefore made search_by_id top-level, so that it is called here.

    Returns: top 5 similar articles

    """
    if request.method == 'POST':
        vector = request.json
        if not (isinstance(vector,list) and all(isinstance(entry, float) for entry in vector)):
            raise ValueError(f"vector payload must be a list of floats. Got {vector}")
        id_ = vector_to_id(vector=vector,articles=articles)
        return  search_by_id_default(id_)

if __name__ == "__main__":
    app.run(debug=True)

#search by id (default: top 5 results) - Social Computing Goes Mobile
# curl http://127.0.0.1:5000/search/43978/

#search by vector -  Government IT Spending In 2008
#curl -d '[0.1426, 0.1087, 0.0609, -0.0799, 0.2368, -0.0665, -0.0103, 0.023700000000000002, 0.054700000000000006, 0.0108, 0.0918, 0.14300000000000002, -0.0499, 0.1612, -0.111, -0.006500000000000001, 0.0791, -0.321, 0.11220000000000001, 0.19490000000000002, -0.1947, 0.1386, -0.08650000000000001, -0.16570000000000001, -0.0912, 0.0449, -0.1149, 0.33180000000000004, -0.15960000000000002, -0.1577, -0.0631, -0.1449, -0.06330000000000001, 0.16490000000000002, -0.1836, 0.1019, -0.044500000000000005, -0.2058, 0.1018, -0.0723, -0.1563, 0.1404, -0.0522, -0.0473, 0.0291, -0.10940000000000001, 0.010700000000000001, 0.0055000000000000005, 0.11080000000000001, -0.039, -0.0026000000000000003, 0.1086, 0.20400000000000001, -0.1131, 0.046700000000000005, -0.2621, -0.0183, -0.084, -0.0816, 0.035500000000000004, 0.047900000000000005, 0.0965, 0.0046, -0.09970000000000001, -0.1602, 0.23, 0.1, -0.0699, 0.3356, -0.0131, 0.0429, -0.32630000000000003, 0.09, -0.3865, 0.25780000000000003, -0.2233, 0.0658, -0.0507, 0.1694, 0.1794, -0.016900000000000002, 0.0548, 0.0507, 0.0453, -0.11570000000000001, 0.0489, 0.06670000000000001, 0.0228, 0.0018000000000000002, -0.0898, 0.0081, -0.1842, -0.0882, 0.3491, -0.1298, -0.125, 0.015700000000000002, -0.19, -0.0873, 0.0756]' -H "Content-Type: application/json" -X POST http://127.0.0.1:5000/search/