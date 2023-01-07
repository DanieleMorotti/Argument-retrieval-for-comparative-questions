import pandas as pd

from .Pipeline import Pipeline


class CombinationPipeline(Pipeline):

    def __init__(self, results_dir, model_name, sparse_index,
                dense_index):
        '''
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


    def __read_topics(self, path_qrels):
        '''
            It returns an ordered list with the ids of the topics.
        '''
        topics = pd.read_csv(path_qrels, sep=" ", 
                            names=['topic', 'Q0', 'docid', 'score'], usecols=[0])
        
        return sorted(topics['topic'].unique())


    def __combine_scores(self, sparse_scores, dense_scores, topics):
        '''
            It combines the scores of the dense and sparse indexes for each topic
            and it returns a dictionary with a new list of ids and scores where in
            the first position there is the most relevant document.
        '''
        rerank = {}

        for key in sparse_scores.keys():
            sparse = {docid:score*0.4 for docid, score in sparse_scores[key]}
            dense = {docid:score*0.6 for docid, score in dense_scores[key]}

            common = sparse.keys() & dense.keys()
            new_scores = {ckey: sparse[ckey]+dense[ckey] for ckey in common}
            new_scores = dict(sorted(new_scores.items(), key=lambda x: x[1], reverse=True))
            
            topic_id = topics[int(key)]
            rerank[topic_id] = {'ids': list(new_scores.keys()), 'scores': list(new_scores.values())}
        return rerank
        

    def compute_results(self, path_qrels, questions_list, corpus_df, k=40):
        '''
            It retrieves the most relevant documents both from sparse and dense
            indexes, it combines the scores and it rerak the documents.
            Parameters:
                - path_qrels: str
                    The path where to find the qrels.
                - questions_list: list[str]
                    A list of the questions (titles).
                - corpus_df: pd.DataFrame
                    The corpus from which we had to retrieve the document ids given
                    the indices of the results of the dense search.
                - k: int
                    The number of documents to retrieve from each index

            Returns:
                It returns the re-ranked list of the documents for each topic and
                it saves the trec file with the results.
        '''
        # Save the .qrels file path to use it afterwards
        self._Pipeline__check_filepath(path_qrels)
        self.path_qrels = path_qrels

        topics = self.__read_topics(path_qrels)
        # Use only the topics that we have in the title list
        queries = [questions_list[ind-1] for ind in topics]

        # Retrieval Algorithm
        sparse_results = self.sparse_index.search(queries, k=k)
        dense_results = self.dense_index.search(queries, k=k, corpus_df=corpus_df)

        comp_rank = self.__combine_scores(sparse_results, dense_results, topics)

        trec_data = self.save_trec(comp_rank, save=True)





        
