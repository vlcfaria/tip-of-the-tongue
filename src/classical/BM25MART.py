from Experiment import Experiment, parse_arguments
import pyterrier as pt
from helper.jsonlHandler import iter_jsonl, transform_fields
import xgboost as xgb
import numpy as np
import pandas as pd

class BM25MART(Experiment):
    '''BM25 with LambdaMART'''
    def __init__(self, index_path: str, topics_path: str, qrels_path: str, corpus_path: str= '', testing=True):
        super().__init__(index_path, corpus_path)

        self.docindex = self.index.getDocumentIndex()
        self.name = 'bm25MART'

        #Defining the initial retrieval process --
        initial_retrieval = pt.terrier.Retriever(self.index, wmodel='BM25')
                
        #Feature engineering --
        bm25_features = pt.terrier.Retriever(self.index, wmodel='BM25')

        dph = pt.terrier.Retriever(self.index, wmodel='DPH')
        pl2 = pt.terrier.Retriever(self.index, wmodel='PL2')
        dfr = pt.terrier.Retriever(self.index, wmodel='DFR_BM25')
        dlh = pt.terrier.Retriever(self.index, wmodel='DLH')
        sdm = pt.rewrite.SequentialDependence() >> bm25_features

        features_pipe = (
            dph ** pl2 ** #Other retrieval models
            initial_retrieval ** #Also use initial retrieval signal
            dfr ** dlh ** # ADD NEW MODELS
            sdm
        )

        #Establishing LTR --
        #this configures xgb as LambdaMART
        lmart = xgb.XGBRanker(
            objective='rank:ndcg',
            eval_metric=['ndcg@10'],
            learning_rate=0.05,
            n_estimators=2000, #High amount, apply early stopping to prevent overfitting
            early_stopping_rounds=50, #Stop after 50 rounds of ndcg not improving
            max_depth=6,
            subsample=0.8, #Fraction of training data sampled for each boosting round
            colsample_bytree=0.8, #Fraction of features randomly sampled for each boosting round
            tree_method='hist',
            random_state=42 # for reproducibility
        )
        lambdaMART = pt.ltr.apply_learned_model(lmart, form='ltr')
        
        #Building the full pipeline -- 
        # Tokenize -> Initial retrieval -> Build features -> LambdaMART
        self.search_pipeline = pt.rewrite.tokenise() >> initial_retrieval >> features_pipe >> lambdaMART
        self.search_pipeline.compile()

        #Training -- 
        qrels = pd.read_csv(qrels_path, 
                            dtype={'qid': 'object', 'docno': 'object', 'relevance': 'int64'})
        topics = pd.read_csv(topics_path, 
                             dtype={'qid': 'object', 'query': 'object'})
        
        #Use this for general testing! Divides the dataset more, but allows to see our nDCG
        if testing:
            train_topics, valid_topics, test_topics = np.split( #shuffle topics
                topics.sample(frac=1, random_state=43),
                [int(.6*len(topics)), int(.8*len(topics))]
            )
        else:
            #Use this for submitting, so we can train on a larger dataset
            train_topics, valid_topics = np.split( #shuffle topics
                topics.sample(frac=1, random_state=43),
                [int(.8*len(topics))]
            )


        self.search_pipeline.fit(train_topics, qrels, valid_topics, qrels)
        print("Feature importances:")
        print(lmart.feature_importances_)

        if testing:
            ans = pt.Experiment([self.search_pipeline], test_topics, qrels, 
                                eval_metrics=['recip_rank', 'ndcg_cut', 'recall'], 
                                names=[self.name])
            print("Benchmark on test topics:")
            print(ans)
            
    def get_index(self, index_path: str):
        return pt.IndexFactory.of(f"{index_path}/data.properties")
    
    def build_index(self, index_path: str, corpus_path: str):
        #Index fields
        return (pt.IterDictIndexer(index_path, meta={'docno': 20}, 
                            text_attrs=['title', 'text', 'keywords'], 
                            fields=True, blocks=True, threads=8)
                .index(iter_jsonl(corpus_path, transform_fields)))


#Example usage
if __name__ == '__main__':
    args = parse_arguments()
    bm = BM25MART(args.index_path, args.experiment_name, args.corpus_path)

    #Benchmark it
    if args.qrels:
        bm.benchmark(args.queries_path, args.qrels, args.out_dir)
    else:
        bm.results_tests(args.queries_path, args.out_dir)