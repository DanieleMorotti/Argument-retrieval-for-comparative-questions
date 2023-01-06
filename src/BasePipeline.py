import pandas as pd
import os

from evaluate_qrels import compute_ndcg

class BasePipeline:

    def __init__(self, results_path, model_name):
        self.results_path = results_path 
        self.model_name = model_name

    
    def __check_filepath(self, filepaths):
        if not isinstance(filepaths, list):
            filepaths = [filepaths]

        for path in filepaths:
            assert os.path.exists(path), "ERROR: wrong file path"


    def evaluate(self, qrel_path, res_path, depth, save=False):
        self.__check_filepath([qrel_path, res_path])
        mean_scores = compute_ndcg(qrel_path, res_path, depth)

        if save:
            mean_scores.to_csv(self.results_path+'ndcg.csv', sep=" ", index=None)
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
            self.res.to_csv(self.results_path+'results.qrels', sep=" ", header=None, index=None)
        
        return self.res
