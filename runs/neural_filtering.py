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
    for c, sample in tqdm(enumerate(data)):
        if not "image" in sample or "VG_100K" in sample["image"]:
            continue
        
        conv = sample["conversations"]
        if not "id" in sample:
            sample["id"] = c
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
            doc = re.sub(r"the image (showcases|features|captures|depicts|shows|presents) ", "", doc)
            
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
    parser.add_argument("--data_paths", nargs="+", type=str, default=["data/vid2r/llava_v1_5_mix665k.json", "data/vid2r/lvis_instruct4v_220k.json"])
    parser.add_argument("--colbert_ckpt", type=str, default="ckpts/colbertv2.0")
    parser.add_argument("--save_path", type=str, default="data/vid2r/vid2r_wo_conversion.json")
    parser.add_argument("--n_bits", type=int, default=2)
    parser.add_argument("--n_ranks", type=int, default=4)
    parser.add_argument("--n_cands", type=int, default=3)
    parser.add_argument("--drop_len", type=int, default=30)
    parser.add_argument("--save_negative", action="store_true")
    args = parser.parse_args()
    
    # Load data
    data = []
    for data_path in args.data_paths:
        ext = data_path.split(".")[-1]
        if ext=="json":
            with open(data_path, "r") as fin:
                data.extend(json.load(fin))
        elif ext=="jsonl":
            data.extend(load_jsonl(data_path))
    
    sub_data, doc_list = filter_1st_stage(data, args)

    collection = Collection(data=doc_list)
    
    print(f"Loaded {len(collection):,} passages")
        
    # Indexing
    index_name = f"filtering2.nbits={args.n_bits}"
    with Run().context(RunConfig(nranks=args.n_ranks, experiment="filtering2")):
        config = ColBERTConfig(nbits=args.n_bits, doc_maxlen=256, query_maxlen=64)
        indexer = Indexer(checkpoint=args.colbert_ckpt, config=config)
        indexer.index(name=index_name, collection=collection, overwrite=True)
    print("Indexing is Finished!")
    
    # Search
    with Run().context(RunConfig(experiment="filtering2")):
        searcher = Searcher(index=index_name, config=config, collection=collection)
    
    filtered = []
    for sample in tqdm(sub_data):
        results = searcher.search(sample["question"], k=args.n_cands)
        passages = [searcher.collection[p_id] for p_id in results[0]] # passage_ids
        if not sample["document"] in passages:
            filtered.append(sample)
            
    print(f"Filtered: {len(sub_data)} -> {len(filtered)}.")
    
    # sub_data -> filtered
    # Align keys
    key2img = {}
    for sample in sub_data:
        if sample["image"] in key2img:
            key2img[sample["image"]].append(sample)
        else:
            key2img[sample["image"]] = [sample]

    for k, sample in enumerate(sub_data):
        ids = [n["id"].strip() for n in key2img[sample["image"]]]
        if len(set(ids))!=len(ids):
            sub_data[k]["id"] = f"{sample['id']}_{random.choice(alpha)}{random.randint(0,9)}"
    
    keyimg_dict = {}
    for k, v in key2img.items():
        keyimg_dict[k] = [d.copy() for d in v]
    
    if args.save_negative:
        # Add negative samples
        for i in range(len(sub_data)):
            negs = keyimg_dict[sub_data[i]["image"]]
            sub_data[i]["negs"] = [ n for n in negs if sub_data[i]["question"]!=n["question"]]
        
    with open(args.save_path, "w") as fout:
        json.dump(sub_data, fout, indent=2, ensure_ascii=False)