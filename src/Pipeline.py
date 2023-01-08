import pandas as pd
import os, re

from .evaluate_qrels import compute_ndcg


class Pipeline:

    def __init__(self, results_dir, model_name):
        '''
            Parameters:
                - results_dir: str
                    The directory where to store the .qrels files and ndcg.
                - model_name: str
                    A name that will represent the model in the .qrels file and
                    in the file names.
        '''
        self.results_dir = results_dir 
        self.model_name = model_name

    
    def __check_filepath(self, filepaths):
        '''
            It checks that a path, or a list of paths, exist.
        '''
        if not isinstance(filepaths, list):
            filepaths = [filepaths]

        for path in filepaths:
            assert os.path.exists(path), "ERROR: wrong file path"


    def __read_topics(self, path_qrels):
        '''
            It returns an ordered list with the unique ids of topics found in 
            the specified path.
        '''
        topics = pd.read_csv(path_qrels, sep=" ", 
                            names=['topic', 'Q0', 'docid', 'score'], usecols=[0])
        
        return sorted(topics['topic'].unique())

    
    def __clean_queries(self, queries):
        '''
            Given a list of strings it returns the clean version of the strings.
        '''
        assert isinstance(queries, list)
        new_queries = []
        for q in queries:
            clean = re.sub("[!'?=.,;:-]", " ", q)
            clean = re.sub(' +',' ', clean)
            new_queries.append(clean)
        return new_queries

    
    def __create_scores_dict(self, scores, topics):
        '''
            It creates the dictionary necessary for creating the trec file.
            The 'scores' dictionary needs to have the following structure:
                {'num': [(docid, score), ...], 'num2':}
            that is the value returned by the DocumentsIndex class.
            It returns a new dictionary where for each topic it contains an 'ids'
            list of document ids and a 'scores' list for retrieval scores.

            Parameters:
                - scores: dict
                    The previously explained dict.
                - topics: list[int]
                    The integers that represent the topics.
        '''
        data = {}

        for key in scores.keys():
            new_dict = {docid: score for docid, score in scores[key]}
            
            topic_id = topics[int(key)]

            data[topic_id] = {'ids': list(new_dict.keys()), 
                                'scores': list(new_dict.values())}
        return data
        

    def evaluate(self, qrel_paths, res_path, depth, save=False):
        '''
            It calls the script to evaluate the pipeline and it retrieves the ndcg
            score in a file or as variable.
            Parameters:
                - qrel_paths: str | list
                    The path where to find the .qrels file to consider or a list
                    of qrel file paths.
                - res_path: str
                    The path where the file with the results is saved (in trec format).
                - depth: int
                    The number to consider for the ndcg.
                - save: bool
                    If a file with the ndcg will be saved or the function should
                    just return the variable with the data.

            Returns:
                It returns a dictionary with the mean of the nDCG scores for each
                qrel file.
        '''
        if not isinstance(qrel_paths, list):
            qrel_paths = [qrel_paths]
            
        self.__check_filepath([*qrel_paths, res_path])
        mean_scores = {}
        for qrel in qrel_paths:
            # Get filename without extension
            qrel_name = os.path.split(qrel)[1].split('.')[0]
            mean_scores[qrel_name] = compute_ndcg(qrel, res_path, depth)
            if save:
                ndcg_path = os.path.join(self.results_dir, 
                                        f"{self.model_name}_{qrel_name}_ndcg{depth}.csv")
                mean_scores[qrel_name].to_csv(ndcg_path, sep=" ", index=None)
        return mean_scores


    def save_trec(self, data, save=False):
        '''
            It receives a dictionary of the type:
                {'topicid': {'ids':[], 'scores':[]}, 'topicid2': {'ids':[], 'scores':[]}}
            
            and it returns and saves the results in a trec file if 'save' is True.
        '''
        self.res = pd.DataFrame(columns=["topic", "Q0", "doc", "rank", "score", "tag"])

        for key in data.keys():
            for i in range(len(data[key]['ids'])):
                # a row is like "1 Q0 clueweb12-en0010-85-29836 1 17.89"
                row = {
                    "topic": str(key),
                    "Q0": "Q0",
                    "doc": data[key]['ids'][i],
                    "rank": i+1,
                    "score": data[key]['scores'][i],
                    "tag": self.model_name
                }
                self.res = self.res.append(row, ignore_index=True)

        if save:
            res_path = os.path.join(self.results_dir, self.model_name + "_results.qrels")
            self.res.to_csv(res_path, sep=" ", header=None, index=None)
        
        return self.res, res_path
