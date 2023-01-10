import pandas as pd

from .Pipeline import Pipeline


class SparsePipeline(Pipeline):

    def __init__(self, results_dir, model_name, sparse_index):
        '''
            It creates an instance of a sparse pipeline.

            Parameters: 
                - results_dir: str
                    The path to the directory where the results will be stored.
                - model_name: str
                    The name to consider for the pipeline.
                - sparse_index: DocumentsIndex
                    The class initialized for the sparse index.
        '''

        super().__init__(results_dir, model_name)
        self.sparse_index = sparse_index
        

    def compute_results(self, path_qrels, questions_list, k=100, clean_query=False,
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
        sparse_results = self.sparse_index.search(queries, k=k)
        sparse_scores = self._Pipeline__create_scores_dict(sparse_results, topics)
        trec_data, res_path = self.save_trec(sparse_scores, save=True)
        
        if evaluate:
            if res_path:
                ndcg = self.evaluate(path_qrels, res_path, 5, False)
                return sparse_scores, ndcg
            else:
                print("ERROR: evaluation set but not provided a valid path for the results.qrels file.")
                return sparse_scores, None
        else:
            return sparse_scores, None
