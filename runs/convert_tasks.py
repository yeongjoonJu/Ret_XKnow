import argparse
import json, re
import random
from tqdm import tqdm
from dataset.base import load_jsonl
from retrievers.colbert.data import Queries, Collection
from retrievers.colbert.infra import Run, RunConfig, ColBERTConfig
from retrievers.colbert import Searcher, Indexer

alpha = ['a','b','c','d','e','f','g','h', 'i', 'j', 'k', "l"]

def is_bad_question(question):
    if "a short description for this region:" in question:
        return True
    if "How many" in question or "What color" in question or "What is the color" in question:
        return True
    
    return False

def filter_1st_stage(data, args):
    miniset = []
    doc_list = []
    for sample in data:
        if not "image" in sample or "VG_100K" in sample["image"]:
            continue
        
        conv = sample["conversations"]
        for t in range(0, len(conv)-1, 2):
            if conv[t]["from"]!="human" or conv[t+1]["from"]!="gpt":
                break
            
            doc = conv[t+1]["value"].strip()
            question = conv[t]["value"].replace("<image>", "").strip()
            
            if len(doc) < args.drop_len or is_bad_question(question):
                continue
            
            # Remove unnecessary prefixes
            doc = doc.lower()
            if doc.split(", ")[0] in ["yes", "no"]:
                doc = ", ".join(doc.split(", ")[1:])
                
            doc = re.sub(r"(certainly|sure)[!.]", "", doc) # Absolutely
            doc = re.sub(r"in (the|this) (image|scene|photo|picture)([,\s])?", "", doc)
            doc = re.sub(r"the image (showcases|features|captures|depicts) ", "", doc)
            
            doc = doc.strip()
            doc = doc[0].upper() + doc[1:]
            
            if "," in question:
                doc = doc.replace(question.split(", ")[0], "")
                                    
            mini_sample = {
                "image": sample["image"],
                "id": f"{sample['id']}_{t}",
                "question": question,
                "document": doc,
            }
            miniset.append(mini_sample)
            doc_list.append(doc)
            
    return miniset, doc_list
        

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/mmqa/mamm_k5.json")
    parser.add_argument("--colbert_ckpt", type=str, default="ckpts/colbertv2.0")
    parser.add_argument("--db_pool", type=str, default="data/wiki/Wiki6M_ver_1_0.jsonl")
    parser.add_argument("--save_path", type=str, default="data/mmqa/mamm_connected_w_wiki.json")
    parser.add_argument("--n_bits", type=int, default=2)
    parser.add_argument("--n_ranks", type=int, default=8)
    parser.add_argument("--n_cands", type=int, default=3)
    args = parser.parse_args()
    
    # Load data
    with open(args.data_path, "r") as fin:
        data = json.load(fin)
    
    wiki_db = load_jsonl(args.db_pool)
    wiki_db = [p['wikipedia_content'].strip() for p in wiki_db]

    collection = Collection(data=wiki_db)
    
    print(f"Loaded {len(collection):,} passages")
        
    # Indexing
    index_name = f"infoseek.nbits={args.n_bits}"
    config = ColBERTConfig(nbits=args.n_bits, doc_maxlen=384, query_maxlen=128)
    with Run().context(RunConfig(nranks=args.n_ranks, experiment="infoseek")):
        indexer = Indexer(checkpoint=args.colbert_ckpt, config=config)
        indexer.index(name=index_name, collection=collection, overwrite=False)
    print("Indexing is finished!")
    
    # Search
    with Run().context(RunConfig(experiment="infoseek")):
        searcher = Searcher(index=index_name, config=config, collection=collection)
        
    for i in tqdm(range(len(data))):
        results = searcher.search(data[i]["document"], k=args.n_cands)
        passages = [searcher.collection[p_id] for p_id in results[0]] # passage_ids
        passages = [".".join(p.split(".")[:3]) for p in passages]
        # data[i]['passages'] = passages
        unified = [passages[0]]
        unified.append(data[i]['document'])
        unified.extend(passages[1:])
        
        data[i]['document'] = " ".join(unified)
                
    with open(args.save_path, "w") as fout:
        json.dump(data, fout, indent=2, ensure_ascii=False)