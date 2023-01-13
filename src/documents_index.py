import shutil, os

import numpy as np

import faiss
from pyserini.search import LuceneSearcher, TctColBertQueryEncoder

class DocumentsIndex:
    allowed_zip_format = ['zip', 'tar', 'tar.gz']

    def __init__(self, index_path, index_type, set_rm3=False, ft=10, fd=10, lam=0.6,
                 set_bm25=False, k=0.9, b=0.4):
        ''' 
            It creates an instance of a sparse or a dense index.

            Parameters:
                - index_path: str
                    The path to the sparse or dense index.
                - index_type: str
                    The string to discriminate between sparse and dense, the
                    possible values are "sparse" or "dense". If another value
                    occurs the index is set to sparse.
                - set_rm3: bool
                    To activate the rm3 query expansion on a sparse index.
                - ft: int
                    The feedback terms to add to the query (only if sparse index
                    and set_rm3 is True).
                - fd: int
                    The number of feedback documents to consider (only if sparse
                    index and set_rm3 is True).
                - lam: float
                    It's the importance given to the original query with respect 
                    to the expansion.
                - set_bm25: bool
                    If you want to set custom parameters for the BM25 score of the
                    sparse index.
                - k: float
                    The value for giving more importance to the frequency of the 
                    terms (only if sparse index and set_bm25 is True).
                - b: float
                    The value for giving more importance to the length of the
                    documents (only if sparse index and set_bm25 is True).
        '''
        # If the index exists we can proceed
        if self.__path_analysis(index_path):
            self.index_path = index_path
            self.index_type = index_type if index_type in ['dense', 'sparse'] else 'sparse'
            self.index = None

            # Only for the dense index
            if self.index_type == 'dense':
                self.enc_name = "castorini/tct_colbert-v2-hnp-msmarco"
                self.encoder = None

            self.__load_index()

            # Set rm3 and bm25 custom parameters for the sparse index 
            if self.index_type == 'sparse':
                if set_rm3:
                    self.index.set_rm3(ft, fd, lam)
                if set_bm25:
                    self.index.set_bm25(k, b)
        

    def __path_analysis(self, path: str):
        '''
            It checks if the file exists and if it is a zip it unpacks it.
            
            Parameters:
                - path: str
                    The path of the index.
            
            Returns:
                - bool
                It returns False if an error occurs, otherwise True.
        '''
        if not os.path.exists(path):
            print(f"ERROR: the index doesn't exist at the following path: {path}")
            return False
        else:
            _, ext = os.path.splitext(path)
            # Unpack the file
            if ext in self.allowed_zip_format:
                print("WARNING: the input path is of a compressed file, now it will be unpacked.")
                shutil.unpack_archive(path)

        return True


    def __load_index(self):
        '''
            It actually loads the index into a variable, and the encoder
            in the case of a dense index.
        '''
        print(f"Loading the {self.index_type} index file ...")
        if self.index_type == "sparse":
            self.index = LuceneSearcher(self.index_path)
        else:
            self.index = faiss.read_index(self.index_path)
            print(f"Loading the encoder {self.enc_name} ...")
            self.encoder = TctColBertQueryEncoder(self.enc_name)

        print("\nThe process is finished correctly!\n")
                
    
    def __index_to_docid(self, indices, corpus_df):
        '''
            It converts indices to document ids (for dense index).
            E.g. 
                indices = [1, 12] -> ['clueweb12-sdajsd__1', 'clueweb12-sdkjda__12']
                
            Parameters: 
                - indices: list[int]
                    A list of indices of the elements we want to retrieve
                    from the dataframe.
                - corpus_df: pd.DataFrame
                    The dataframe of the corpus.
            
            Returns:
                - list[str]
                    It returns a list of strings that represent the ids
                    of the documents.
        '''
        indices = indices.tolist()
        for ind in range(len(indices)):
            indices[ind] = corpus_df.iloc[indices[ind]]['id'].tolist()
        return indices
    

    def search(self, query: str, k: int=40, corpus_df=None, verbose=False):
        ''' 
            If the query is a single string it returns a list of tuples (id, score),
            otherwise it returns a dictionary, the numbers with the position of 
            the query in the array as keys. For each query you will find the same 
            array as for the single query. 
            E.g.
                query = "Dogs or cats?"
                res = [(doc_id1, score1), (doc_id2, score2), ...]

                query = ["Dogs or cats?", "Coke or Pepsi?"]
                res = {'0':[(doc_id1, score1),...], '1': [(doc_id1, score1,...)]}
            
            Parameters:
                - query: str | list[str]
                    A string or a list of strings.
                - k: int
                    The number of documents to retrieve from the index.
                - corpus_df: pd.DataFrame
                    If the index is dense we need it to convert the indices to the
                    document ids.
                - verbose: bool
                    It True then the results of the search will be printed.

            Returns: 
                - dict
                    A dictionary where for each key, that represents the index of the question
                    in the input query list, there is a list of tuples with the document id and
                    the relative score, ordered by the score. In case of a single string passed
                    the dictionary has the default '0' key.
        '''
        if self.index_type == 'sparse':
            if isinstance(query, list):
                # Run batch search if we have a list of queries
                hits = self.index.batch_search(query, [str(i) for i in range(len(query))], k=k)
                for key in hits.keys():
                    hits[key] = [(el.docid, el.score) for el in hits[key]]
            else:
                hits = self.index.search(query, k=k)
                hits = {'0': [(el.docid, el.score) for el in hits]}
        else:
            if corpus_df is None:
                print("ERROR: corpus needed for the dense index in order to convert the indices to the document ids.")
                return None
            if isinstance(query, list):
                embeddings = []
                # Create the embedding for each query and stack them to create a batch
                for q in query:
                    query_enc = self.encoder.encode(q)
                    query_enc = np.expand_dims(query_enc, axis=0)
                    embeddings.append(query_enc)

                # Stack the embeddings of the queries to pass a batch (len(embeddings), 768)
                distances, indices = self.index.search(np.concatenate(embeddings), k)
                # Convert indices to document ids
                indices = self.__index_to_docid(indices, corpus_df)

                hits = {str(i): list(zip(indices[i], distances[i])) 
                            for i in range(len(query))}
            else:
                query_enc = self.encoder.encode(query)
                query_enc = np.expand_dims(query_enc, axis=0)
                distances, indices = self.index.search(query_enc, k)
                # Convert indices to document ids
                indices = self.__index_to_docid(indices, corpus_df)
                hits = {'0': list(zip(indices[0], distances[0]))}

        if verbose:
            self.__print_results(hits)
        return hits


    def __print_results(self, results):
        '''
            It prints the results of the search given the id and the score ordered
            considering the score.
        '''
        if isinstance(results, dict):
            for key in results.keys():
                print(f"{int(key)+1}. For the query {key}:")
                for i, (ind, score) in enumerate(results[key]):
                    print(f'{i+1:2} {ind:4} {score:.4f}')
        else:
            for i, (ind, score) in enumerate(results):
                    print(f'{i+1:2} {ind:4} {score:.4f}')