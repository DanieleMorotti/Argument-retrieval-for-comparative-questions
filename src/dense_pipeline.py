import pandas as pd

from .pipeline import Pipeline


class DensePipeline(Pipeline):

    def __init__(self, results_dir, model_name, dense_index):
        '''
            It creates an instance of a dense pipeline.

            Parameters: 
                - results_dir: str
                    The path to the directory where the results will be stored.
                - model_name: str
                    The name to consider for the pipeline.
                - dense_index: DocumentsIndex
                    The class initialized for the dense index.
        '''

        super().__init__(results_dir, model_name)
        self.dense_index = dense_index
       

    def compute_results(self, path_qrels, questions_list, corpus_df, k=100, 
                        clean_query=False, evaluate=False):
        '''
            It retrieves the most relevant documents from dense index, the
            ranking is done considering the inner product between the query and 
            the document embeddings.

            Parameters:
                - path_qrels: str
                    The path where to find the qrels, one file or a list of paths.
                - questions_list: list[str]
                    A list of the questions (titles).
                - corpus_df: pd.DataFrame
                    The corpus from which we had to retrieve the document ids given
                    the indices of the results of the dense search.
                - k: int
                    The number of documents to retrieve from each index
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
        dense_results = self.dense_index.search(queries, k=k, corpus_df=corpus_df)
        dense_scores = self._Pipeline__create_scores_dict(dense_results, topics)
        trec_data, res_path = self.save_trec(dense_scores, save=True)

        if evaluate:
            if res_path:
                ndcg = self.evaluate(path_qrels, res_path, 5, False)
                return dense_scores, ndcg
            else:
                print("ERROR: evaluation set but not provided a valid path for the results.qrels file.")
                return dense_scores, None
        else:
            return dense_scores, None
       
