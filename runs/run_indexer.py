import argparse
from retrievers.indexing import XknowIndexer
from retrievers.colbert.infra import Run, RunConfig, ColBERTConfig
from retrievers.colbert import Indexer
from retrievers.colbert.data import Collection

import os
os.environ["NCCL_BLOCKING_WAIT"] = "1"
os.environ["NCCL_TIMEOUT"] = "7000000"

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--n_bits", type=int, default=2)
    parser.add_argument("--n_ranks", type=int, default=4)
    parser.add_argument("--all_blocks_file", type=str, default="data/okvqa/all_blocks.txt")
    parser.add_argument("--colbert_ckpt", type=str, default="ckpts/colbertv2.0")
    parser.add_argument("--xknow_ckpt", type=str, default=None)
    parser.add_argument("--max_doc_len", type=int, default=384)
    parser.add_argument("--dataset_name", type=str, default="okvqa", help="okvqa|okvqa_gs")
    args = parser.parse_args()
    
    if args.dataset_name=="okvqa":
        from dataset.okvqa import get_collection
        collection = get_collection(args.all_blocks_file)
        
    elif args.dataset_name=="okvqa_gs":
        from dataset.vqa_ret import get_collection
        collection = get_collection(args.all_blocks_file)
    
    elif args.dataset_name=="fvqa":
        from dataset.fvqa import get_collection
        collection, _ = get_collection(args.all_blocks_file)
                
    elif args.dataset_name=="infoseek":
        from dataset.infoseek import get_collection
        collection = get_collection(args.all_blocks_file)
        
    elif args.dataset_name=="aokvqa":
        from dataset.aokvqa import parse_pairs
        _, pid2passage = parse_pairs(args.all_blocks_file)
        collection = Collection(data=list(pid2passage.values()))
        
    elif args.dataset_name in ["vid2r", "vl_ict"]:
        from dataset.vid2r import parse_pairs
        _, pid2passage = parse_pairs(args.all_blocks_file)
        collection = Collection(data=list(pid2passage.values()))
        
    elif args.dataset_name=="remuq":
        import pandas as pd
        _pid2passage = pd.read_csv(args.all_blocks_file)
        _pid2passage = _pid2passage.to_dict()
        pid2passage = {}
        for kid, text in zip(_pid2passage['kid'].values(), _pid2passage['text'].values()):
            pid2passage[(kid)] = text
        passage_ids = list(pid2passage.keys())
        collection = Collection(data=list(pid2passage.values()))
    else:
        raise NotImplementedError
    
    
    
    index_name = f"{args.exp_name}.nbits={args.n_bits}"
    with Run().context(RunConfig(nranks=args.n_ranks, experiment=args.exp_name)):
        config = ColBERTConfig(nbits=args.n_bits, doc_maxlen=args.max_doc_len, query_maxlen=64)
        if args.xknow_ckpt is not None:
            indexer = XknowIndexer(checkpoint=args.xknow_ckpt, config=config)
        else:
            indexer = Indexer(checkpoint=args.colbert_ckpt, config=config)
            
        indexer.index(name=index_name, collection=collection, overwrite=True)
        
    print("Complete Indexing!")