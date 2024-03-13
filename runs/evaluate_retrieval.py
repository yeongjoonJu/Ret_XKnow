import argparse
import json, os
from io import BytesIO
import base64
# For colbert
from retrievers.colbert.infra import Run, RunConfig, ColBERTConfig
from retrievers.colbert import Searcher
# For Ret-XKnow
from retrievers.indexing import XknowSearcher
from dataset.base import load_jsonl
import pytrec_eval
import numpy as np
import random


def search(queries, collection, args, images=None):
    nbits = int(args.index_name.split("nbits=")[1])
    config = ColBERTConfig(nbits=nbits, doc_maxlen=384, query_maxlen=64)
    
    queries = {key: queries[key] for key in sorted(queries)}
    if args.image_dir is not None:
        images = {key: images[key] for key in sorted(images)}

    # Search
    print("Searching top-100...")
    with Run().context(RunConfig(experiment=args.index_name.split(".")[0])):
        if args.xknow_ckpt is not None:
            searcher = XknowSearcher(index=args.index_name, checkpoint=args.xknow_ckpt, config=config, collection=collection)
            ranking = searcher.search_all(queries=queries, images=images, k=100,)
        else:
            searcher = Searcher(index=args.index_name, checkpoint=args.colbert_ckpt, config=config, collection=collection)
            ranking = searcher.search_all(queries=queries, k=100,)
    
    data = {}
    for items in ranking.flat_ranking:
        qid = items[0]
        if qid in data:
            data[qid]["I"].append(items[1])
            data[qid]["D"].append(items[-1])
        else:
            data[qid] = {"I": [items[1]], "D": [items[-1]]}

    return data

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--colbert_ckpt", type=str, default="ckpts/colbertv2.0")
    parser.add_argument("--index_name", type=str, default="okvqa.nbits=2")
    parser.add_argument("--anno_file", type=str, default="data/okvqa/test2014_pairs_cap_combine_sum.txt")
    parser.add_argument("--all_blocks_file", type=str, default="data/okvqa/all_blocks.txt")
    parser.add_argument("--save_path", type=str, default="results/eval_okvqa_xknow_clip.json")
    parser.add_argument("--image_dir", type=str, default=None, help="data/wiki/wikipedia_images_full")
    parser.add_argument("--xknow_ckpt", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="okvqa", help="okvqa|remuq|okvqa_gs|infoseek")
    args = parser.parse_args()
    
    queries = {}
    images = {}
    if args.dataset_name=="okvqa":
        from dataset.okvqa import DynamicEval, get_collection, parse_pairs
        # Load eval data
        dynamic_eval = DynamicEval(
            "data/okvqa/mscoco_val2014_annotations.json",
            "data/okvqa/OpenEnded_mscoco_val2014_questions.json",
            passage_id_to_line_id_file="data/okvqa/passage_id_to_line_id.json",
            all_blocks_file=args.all_blocks_file
        )
        
        # Passages
        pairs = parse_pairs(args.anno_file)
        for d in pairs:
            queries[d["q_id"]] = d["question"]
            if args.image_dir is not None:
                images[d["q_id"]] = f"{args.image_dir}/{d['image']}"
        
        collection, passage_ids = get_collection(args.all_blocks_file, return_ids=True)
       
    elif args.dataset_name=="okvqa_gs":
        from dataset.vqa_ret import parse_pairs, get_collection, gen_qrels
        
        # Passages
        pairs = parse_pairs(args.anno_file)
        """
        with open("data/webqa/imgs.lineidx", "r") as fp_lineidx:
            lineidx = [int(i.strip()) for i in fp_lineidx.readlines()]
        fp = open("data/webqa/imgs.tsv", "r")
        
        for d in pairs:
            queries[d["q_id"]] = d["question"]
            fp.seek(lineidx[int(d["image"])%10000000])
            imgid, img_base64 = fp.readline().strip().split("\t")
            images[d["q_id"]] = BytesIO(base64.b64decode(img_base64))  
        fp.close()
        """
        for d in pairs:
            queries[d["q_id"]] = d["question"]
            if args.image_dir is not None:
                images[d["q_id"]] = f"{args.image_dir}/{d['image']}.jpg"

        collection, passage_ids, pid2passage = get_collection(args.all_blocks_file, return_ids=True)

    elif args.dataset_name=="infoseek":
        from dataset.infoseek import parse_pairs, get_collection
        from dataset.vqa_ret import gen_qrels

        collection, passage_ids, pid2passage = get_collection(args.all_blocks_file, return_ids=True) # wiki_db
        # kb_mapping = load_jsonl("data/infoseek/oven_entity_test.jsonl") # infoseek_val_withkb
        pairs = parse_pairs(args.anno_file)#, kb_mapping=kb_mapping, wiki_db=wiki_db)
        
        data = []
        not_exist_images = 0
        for d in pairs:
            queries[d["q_id"]] = d["question"]
            if args.image_dir is not None:
                image_path = f"{args.image_dir}/{d['image']}"
                if not os.path.exists(image_path):
                    image_path = ".".join(image_path.split(".")[:-1]) + ".jpg"
                    if not os.path.exists(image_path):
                        not_exist_images+=1
                images[d["q_id"]] = image_path

        print("Not exists images:", not_exist_images, "/", len(pairs))

    # Search
    data = search(queries, collection, images=images, args=args)

    eval_metrics = {}
    for ret_rank in [1, 5, 10, 20, 50, 100]:
        print("Evaluating Rank", ret_rank)
        # qids: question ids
        # I: retrieved_ids
        # D: scores
        qids = list(data.keys())
        I, D = [], []
        for qid in qids:
            I.append(data[qid]["I"][:ret_rank])
            D.append(data[qid]["D"][:ret_rank])
        
        # query relevance
        if args.dataset_name=="okvqa":
            qrels = dynamic_eval.gen_qrels(qids, I, passage_ids)
        else:
            passage_ids = [str(k) for k in passage_ids]
            # query relevance
            # golden_pids = {int(s["q_id"]): str(s["golden_pid"]) for s in pairs}
            qid2answers = {int(s['q_id']): s['answers'] for s in pairs}
            qrels = gen_qrels(qids, I, passage_ids, qid2answers=qid2answers, pid2passage=pid2passage)
        
        run = {}
        for qid, retrieved_ids, scores in zip(qids, I, D):
            run[str(qid)] = {passage_ids[retrieved_id]: float(score) for retrieved_id, score in zip(retrieved_ids, scores)}
        
        if ret_rank==5:
            evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'recip_rank', 'P_5', 'recall_5'})
        elif ret_rank==1:
            evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"P_1"})
        else:
            evaluator = pytrec_eval.RelevanceEvaluator(qrels, {f'recall_{ret_rank}'})

        metrics = evaluator.evaluate(run)
        
        if ret_rank==5:
            eval_metrics['MRR'] = np.average([v['recip_rank'] for v in metrics.values()])
            eval_metrics["P@5"] = np.average([v['P_5'] for v in metrics.values()])
            eval_metrics["R@5"] = np.average([v['recall_5'] for v in metrics.values()])
        elif ret_rank==1:
            eval_metrics["P@1"] = np.average([v['P_1'] for v in metrics.values()])
        else:
            eval_metrics[f"R@{ret_rank}"] = np.average([v[f'recall_{ret_rank}'] for v in metrics.values()])
        
    for k, v in eval_metrics.items():
        print(f"{k}: {v}")
    
    with open(args.save_path, "w") as fout:
        json.dump(eval_metrics, fout, ensure_ascii=False, indent=2)