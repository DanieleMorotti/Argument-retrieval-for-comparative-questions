def print_ndcg(ndcg_scores):
    '''
        It prints the nDCG score for each key in the input.

        Parameters: 
            - ndcg_scores: dict
                The dictionary that contains for each .qrels file
                the mean nDCG.
    '''
    if ndcg_scores and isinstance(ndcg_scores, dict):
        for key, ndcg in ndcg_scores.items():
            print(f"The nDCG for {key} qrel is:")
            print(ndcg, '\n')
    else:
        print("nDCG has not been computed, set 'evaluate=True' to compute it.")


# Retrieve the urls from the corpus given the results of the search on the index
def retrieve_docs_ranked(corpus, hits, k=10):
    '''
        It returns a list of ids and urls retrieved from the corpus
        using the ids within hits.

        Parameters:
            - corpus: pd.DataFrame
                The corpus from which we want to retrieve the urls.
            - hits: dict
                The dictionary that has ids and scores for a specific
                topic.
            - k: int
                The number of elements to returns. 
    '''
    urls = []
    for el in hits['ids'][:k]:
        urls.append(corpus[corpus['id'] == el]['chatNoirUrl'].item())
    ids = [val for val in hits['ids'][:k]]
    return ids, urls