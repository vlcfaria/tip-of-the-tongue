#We dont use the pyterrier framework here
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" #same order as nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Script to index a tsv collection using colBERT.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        'index_root',
        type=str,
        help='Directory of the index root'
    )

    parser.add_argument(
        'corpus_path',
        type=str,
        help='Path to the .tsv corpus'
    )

    args = parser.parse_args()
    
    return args

checkpoint = 'colbert-ir/colbertv2.0'

if __name__=='__main__':
    args = parse_arguments()
    with Run().context(RunConfig(nranks=1, experiment="tot", gpus=1, root=args.index_root)): #nranks -> gpus to use

        config = ColBERTConfig(
            nbits=2,
            gpus=1,
            bsize=3000,
        )
        indexer = Indexer(checkpoint=checkpoint, config=config)
        indexer.index(name="tot.nbits=2", collection=args.corpus_path)
