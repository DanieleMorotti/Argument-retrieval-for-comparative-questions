import pandas as pd
from collections import defaultdict

from .pipeline import Pipeline


class HybridPipeline(Pipeline):

    def __init__(self, results_dir, model_name, sparse_index, dense_index):
        ''' 
            It creates the instance for a hybrid pipeline.

            Parameters: 
                - results_dir: str
                    The path to the directory where the results will be stored.
                - model_name: str
                    The name to consider for the pipeline.
                - sparse_index: DocumentsIndex
                    The class initialized for the sparse index.
                - dense_index: DocumentsIndex
                    The class initialized for the dense index.
        '''
        super().__init__(results_dir, model_name)
        self.sparse_index = sparse_index
        self.dense_index = dense_index
    

    def __hybrid_rerank(self, dense_results, sparse_results, alpha, k, normalization=False, weight_on_dense=False):
        '''
            It takes as input the dense and sparse rankings and it returns a new
            dictionary with a re-ranking made considering both scores and an alpha
            factor to weight more one of the two.

            Parameters:
                - dense_results: dict
                    A dictionary that contains the dense index results for each 
                    query, i.e. a list of tuples (docid, score) sorted considering
                    the score.
                - sparse_results: dict
                    A dictionary that contains the sparse index results for each 
                    query, i.e. a list of tuples (docid, score) sorted considering
                    the score.
                - alpha: float
                    A factor to multiply the dense or sparse score. Default behaviour,
                    it is used for multiplying the sparse score.
                - k: int
                    The number of documents to retrieve for each query.
                - normalization:: bool
                    If True the scores will be normalized.
                - weight_on_dense: bool
                    If True it applies alpha to the dense score.
            
            Returns:
                - dict
                    A dictionary that contains for each topic a list of tuples (docid, score)
                    sorted in a way that the first element has the largest score.
        '''
        hybrid_result = defaultdict(list)
        # Rerank for each topic
        for key in dense_results.keys():
            dense_hits = {docid: score for docid, score in dense_results[key]}
            sparse_hits = {docid: score for docid, score in sparse_results[key]}
            min_dense_score = min(dense_hits.values()) if len(dense_hits) > 0 else 0
            max_dense_score = max(dense_hits.values()) if len(dense_hits) > 0 else 1
            min_sparse_score = min(sparse_hits.values()) if len(sparse_hits) > 0 else 0
            max_sparse_score = max(sparse_hits.values()) if len(sparse_hits) > 0 else 1
            for doc in set(dense_hits.keys()) | set(sparse_hits.keys()):
                if doc not in dense_hits:
                    sparse_score = sparse_hits[doc]
                    dense_score = min_dense_score
                elif doc not in sparse_hits:
                    sparse_score = min_sparse_score
                    dense_score = dense_hits[doc]
                else:
                    sparse_score = sparse_hits[doc]
                    dense_score = dense_hits[doc]
                if normalization:
                    sparse_score = (sparse_score - (min_sparse_score + max_sparse_score) / 2) \
                                / (max_sparse_score - min_sparse_score)
                    dense_score = (dense_score - (min_dense_score + max_dense_score) / 2) \
                                / (max_dense_score - min_dense_score)
                score = alpha * sparse_score + dense_score if not weight_on_dense else sparse_score + alpha * dense_score
                hybrid_result[key].append((doc, score))

            hybrid_result[key] = sorted(hybrid_result[key], key=lambda x: x[1], reverse=True)[:k]
        return dict(hybrid_result)


    def compute_results(self, path_qrels, questions_list, corpus_df, k=100, 
                        alpha=0.2, weight_on_dense=False, normalization=False, 
                        clean_query=False, evaluate=False, k_end=40):
        '''
            It retrieves the most relevant documents considering a hybrid score
            between the dense and the sparse indexes.

            Parameters:
                - path_qrels: str
                    The path where to find the qrels, one file or a list of paths.
                - questions_list: list[str]
                    A list of the questions (titles).
                - corpus_df: pd.DataFrame
                    The corpus from which we had to retrieve the document ids given
                    the indices of the results of the dense search.
                - k: int
                    The number of documents to retrieve from each index.
                - alpha: float
                    The number to multiply by one of the 2 scores (sparse score by default).
                - weight_on_dense: bool
                    If the alpha has to be applied to the dense score.
                - normalization: bool
                    If we want to apply normalization of the scores.
                - clean_query: bool
                    If it's True then the queries will be submitted removing punctuation
                    and adjacent whitespaces.
                - evaluate: bool
                    If True then the nDCG@5 will be computed and it will be returned.
                - k_end: int
                    The number of documents to return after re-ranking.

            Returns:
                It returns the list of the most relevant documents for each topic
                and the ndcg if evaluate is True, otherwise None.
        '''
        self._Pipeline__check_filepath(path_qrels)

        # We know that the topics are the same in all the qrels in our case
        if isinstance(path_qrels, list):
            topics = self._Pipeline__read_topics(path_qrels[0])
        else:
            topics = self._Pipeline__read_topics(path_qrels)
            
        # Use only the topics that were selected for the task
        queries = [questions_list[ind-1] for ind in topics]
        if clean_query:
            queries = self._Pipeline__remove_punctuation(queries)

        # Retrieval Algorithm
        # 1. Retrieve documents from both indexes  
        sparse_results = self.sparse_index.search(queries, k=k)
        dense_results = self.dense_index.search(queries, k=k, corpus_df=corpus_df)

        # 2. Re-rank the previous results and creates a correct dictionary for evaluation
        hybrid_results = self.__hybrid_rerank(dense_results, sparse_results, alpha,
                                              k_end, normalization, weight_on_dense)
        hybrid_scores = self._Pipeline__create_scores_dict(hybrid_results, topics)

        trec_data, res_path = self.save_trec(hybrid_scores, save=True)

        if evaluate:
            if res_path:
                ndcg = self.evaluate(path_qrels, res_path, 5, False)
                return hybrid_scores, ndcg
            else:
                print("ERROR: evaluation set but not provided a valid path for the results.qrels file provided.")
                return hybrid_scores, None
        else:
            return hybrid_scores, None
