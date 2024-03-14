import argparse
import json
from io import BytesIO
import base64
# For colbert
from retrievers.colbert.infra import Run, RunConfig, ColBERTConfig
from retrievers.colbert import Searcher
# For Ret-XKnow
from retrievers.indexing import XknowSearcher
from dataset.base import convert_list2dict
from tqdm import tqdm
import random
import re


def has_answers(answers, passage):
    passage = passage.lower()
    for answer in answers:
        if re.search(r'\b{}\b'.format(answer.lower()), passage):
            return True
    return False

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--colbert_ckpt", type=str, default="ckpts/colbertv2.0")
    parser.add_argument("--index_name", type=str, default="okvqa.nbits=2")
    parser.add_argument("--anno_file", type=str, default="data/okvqa/test2014_pairs_cap_combine_sum.txt")
    parser.add_argument("--all_blocks_file", type=str, default="data/okvqa/all_blocks.txt")
    parser.add_argument("--save_path", type=str, default="results/train_hard.json")
    parser.add_argument("--image_dir", type=str, default=None, help="data/wiki/wikipedia_images_full")
    parser.add_argument("--xknow_ckpt", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="okvqa", help="okvqa|remuq|okvqa_gs")
    parser.add_argument("--topk", type=int, default=150)
    args = parser.parse_args()
    
    queries = {}
    images = {}
    if args.dataset_name in ["okvqa_gs", "remuq"]:
        from dataset.vqa_ret import parse_pairs, get_collection, gen_qrels
        with open(args.anno_file, "r") as f:
            samples = json.load(f)
        pairs = []
        ori_samples = []
        for sample in samples:
            q_id = sample["question_id"]
            img_id = sample["img_id"]
            question = sample["question"]
            passages = sample["ctxs"]

            if not passages:
                continue
            pos_passage = random.choice(passages) # if random_choice else passages[0]
            pos_passage = pos_passage['text']
            if "caption" in sample:
                caption = sample["caption"]
                question = question.replace(caption, "").strip()
            
            # image_path = kb_id2img_path(f"Q{img_id}")
            # golden_pid = sample["ctxs"][0]["id"]
            answers = list(sample["answers"].keys())

            pairs.append({
                "question": question, "document": pos_passage, "answers": answers,
                "image": img_id, "q_id": q_id,# "golden_pid": golden_pid
            })
            ori_samples.append(sample)

        if args.dataset_name=="remuq":
            with open("data/webqa/imgs.lineidx", "r") as fp_lineidx:
                lineidx = [int(i.strip()) for i in fp_lineidx.readlines()]
            fp = open("data/webqa/imgs.tsv", "r")
            
            for d in pairs:
                queries[d["q_id"]] = d["question"]
                fp.seek(lineidx[int(d["image"])%10000000])
                imgid, img_base64 = fp.readline().strip().split("\t")
                images[d["q_id"]] = BytesIO(base64.b64decode(img_base64))  
            fp.close()
        else:
            for d in pairs:
                queries[d["q_id"]] = d["question"]
                if args.image_dir is not None:
                    images[d["q_id"]] = f"{args.image_dir}/{d['image']}.jpg"

        collection, passage_ids, pid2passage = get_collection(args.all_blocks_file, return_ids=True)
    else:
        raise NotImplementedError

    nbits = int(args.index_name.split("nbits=")[1])
    config = ColBERTConfig(nbits=nbits, doc_maxlen=384, query_maxlen=64)
    
    queries = {key: queries[key] for key in sorted(queries)}
    if args.image_dir is not None:
        images = {key: images[key] for key in sorted(images)}

    # Search
    print(f"Searching top-{args.topk}...")
    with Run().context(RunConfig(experiment=args.index_name.split(".")[0])):
        if args.xknow_ckpt is not None:
            searcher = XknowSearcher(index=args.index_name, checkpoint=args.xknow_ckpt, config=config, collection=collection)
            ranking = searcher.search_all(queries=queries, images=images, k=args.topk)
        else:
            searcher = Searcher(index=args.index_name, checkpoint=args.colbert_ckpt, config=config, collection=collection)
            ranking = searcher.search_all(queries=queries, k=args.topk)
    
    data = {}
    for items in ranking.flat_ranking:
        qid = items[0]
        if qid in data:
            data[qid]["I"].append(items[1])
            data[qid]["D"].append(items[-1])
        else:
            data[qid] = {"I": [items[1]], "D": [items[-1]]}

    qids = list(data.keys())
    I, D = [], []
    for qid in qids:
        I.append(data[qid]["I"])
        D.append(data[qid]["D"])

    ori_samples = convert_list2dict(ori_samples, set_key="question_id")

    # query relevance
    if args.dataset_name=="okvqa":
        pass
    else:
        passage_ids = [str(k) for k in passage_ids]
        # query relevance
        # golden_pids = {int(s["q_id"]): str(s["golden_pid"]) for s in pairs}
        qid2answers = {int(s['q_id']): s['answers'] for s in pairs}

        qrels = {}
        for question_id, retrieved_ids in tqdm(zip(qids, I), total=len(qids)):
            if question_id not in qrels:
                qrels[str(question_id)] = {'placeholder': 0}

            for retrieved_id in retrieved_ids:
                passage_id = passage_ids[retrieved_id]
                answers = qid2answers[int(question_id)]
                passage = pid2passage[passage_id]

                if has_answers(answers, passage):
                    qrels[str(question_id)][passage_id] = 1
                else:
                    if not "negs" in ori_samples[question_id]:
                        ori_samples[question_id]["negs"] = [passage]
                    else:
                        ori_samples[question_id]["negs"].append(passage)

            if not 'negs' in ori_samples[question_id]:
                ori_samples[question_id]['negs'] = [pid2passage[passage_ids[r_id]] for r_id in retrieved_ids[-32:]]
            if len(ori_samples[question_id]['negs']) < 32:
                ori_samples[question_id]['negs'] += [pid2passage[passage_ids[r_id]] for r_id in retrieved_ids[-1*(32-len(ori_samples[question_id]['negs'])):]]

    listed_samples = []

    for k, v in ori_samples.items():
        listed_samples.append(v)
        listed_samples[-1]["question_id"] = k

    with open(args.save_path, "w") as fout:
        json.dump(listed_samples, fout, ensure_ascii=False, indent=2)