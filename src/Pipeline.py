import pandas as pd
import os

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


    def evaluate(self, qrel_path, res_path, depth, save=False):
        '''
            It calls the script to evaluate the pipeline and it retrieves the ndcg
            score in a file or as variable.
            Parameters:
                - qrel_path: str
                    The path where to find the .qrels file to consider.
                - res_path: str
                    The path where the file with the results is saved (in trec format).
                - depth: int
                    The number to consider for the ndcg.
                - save: bool
                    If a file with the ndcg will be saved or the function should
                    just return the variable with the data.

            Returns:
                It returns the mean of the ndcg scores.
        '''
        self.__check_filepath([qrel_path, res_path])
        mean_scores = compute_ndcg(qrel_path, res_path, depth)

        if save:
            ndcg_path = os.path.join(self.results_dir, self.model_name + "_ndcg.csv")
            mean_scores.to_csv(ndcg_path, sep=" ", index=None)
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
                    "topic": key,
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
        
        return self.res
