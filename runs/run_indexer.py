import argparse
from retrievers.indexing import XknowIndexer
from retrievers.colbert.infra import Run, RunConfig, ColBERTConfig
from retrievers.colbert import Indexer

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
    elif args.dataset_name=="okvqa_gs":
        from dataset.vqa_ret import get_collection
    elif args.dataset_name=="infoseek":
        from dataset.infoseek import get_collection
    else:
        raise NotImplementedError
    
    # Passages
    collection = get_collection(args.all_blocks_file)
    
    index_name = f"{args.exp_name}.nbits={args.n_bits}"
    with Run().context(RunConfig(nranks=args.n_ranks, experiment=args.exp_name)):
        config = ColBERTConfig(nbits=args.n_bits, doc_maxlen=args.max_doc_len, query_maxlen=64)
        if args.xknow_ckpt is not None:
            indexer = XknowIndexer(checkpoint=args.xknow_ckpt, config=config)
        else:
            indexer = Indexer(checkpoint=args.colbert_ckpt, config=config)
            
        indexer.index(name=index_name, collection=collection, overwrite=True)
        
    print("Complete Indexing!")