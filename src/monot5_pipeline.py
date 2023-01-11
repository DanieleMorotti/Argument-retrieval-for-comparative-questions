from typing import Dict, Tuple, List

from pygaggle.rerank.base import Query, Text
from pygaggle.rerank.transformer import MonoT5

import pandas as pd

from .pipeline import Pipeline
from .documents_index import DocumentsIndex


class MonoT5Pipeline(Pipeline):

    def __init__(self, results_dir, model_name, index: DocumentsIndex, monot5_name="castorini/monot5-base-msmarco"):
        '''
            It creates an instance of the pipeline.

            Parameters: 
                - results_dir: str
                    The path to the directory where the results will be stored.
                - model_name: str
                    The name to consider for the pipeline.
                - index: DocumentsIndex
                    The class initialized for the sparse index.
        '''

        super().__init__(results_dir, model_name)
        self.index = index

        self.monot5 = MonoT5(pretrained_model_name_or_path=monot5_name)
        

    def compute_results(self, path_qrels, questions_list, corpus_df: pd.DataFrame, k=100, clean_query=False,
                        evaluate=False):
        '''
            It retrieves the most relevant documents from sparse index, therefore
            the rank is done considering BM25 metric.

            Parameters:
                - path_qrels: str
                    The path where to find the qrels, one file or a list of paths.
                - questions_list: list[str]
                    A list of the questions (titles).
                - k: int
                    The number of documents to retrieve from the index
                - clean_query: bool
                    If it's True then the queries will be submitted removing punctuation
                    and adjacent whitespaces.
                - evaluate: bool
                    If True then the nDCG@5 will be computed and it will be returned.

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
        sparse_results = self.index.search(queries, k=k)

        # Reranking
        reranked_results = self.__reranking(queries, corpus_df, sparse_results)

        scores = self._Pipeline__create_scores_dict(reranked_results, topics)
        trec_data, res_path = self.save_trec(scores, save=True)
        
        if evaluate:
            if res_path:
                ndcg = self.evaluate(path_qrels, res_path, 5, False)
                return scores, ndcg
            else:
                print("ERROR: evaluation set but not provided a valid path for the results.qrels file.")
                return scores, None
        else:
            return scores, None


    def __reranking(self, topics_idx: List[int], queries: List[str], corpus_df: pd.DataFrame, results: Dict[int, List[Tuple[str, float]]]):
        reranked_scores = {}

        for key in results.keys():
            topic_id = topics_idx[int(key)]

            query = Query(queries[topic_id-1])
            texts = [Text(corpus_df[corpus_df.id == doc_id].contents.item(), {"docid": doc_id}, 0) for (doc_id, _) in results[key]]
            reranked = self.monot5.rerank(query, texts)
            
            reranked_scores[key] = [(doc.metadata["docid"], doc.score) for doc in reranked]
        
        return reranked_scores
        