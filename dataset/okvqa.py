import os
import json
import linecache
import re, jsonlines
from collections import defaultdict
from dataset.base import BaseRetrievalDataset, load_jsonl, convert_list2dict
from retrievers.colbert.data import Collection
from dataset.vqa_utils import VQA
import torch
from tqdm import tqdm
import random
from dataset.vqa_ret import parse_pairs as parse_pairs_for_gs

def imageid2path(img_id, split="train|val"):
    assert split in ["train", "val"]
    
    img_id = str(img_id)
    prefix = (12 - len(img_id))*"0"
    image_path = f"coco/{split}2014/COCO_{split}2014_{prefix}{img_id}.jpg"
    
    return image_path
    
def get_collection(filename, return_ids=False):
    with open(filename, "r") as f:
        lines = f.readlines()
    
    passage_ids = []
    passages = []            
    for line in lines:
        entry = json.loads(line.strip())
        passage_ids.append(entry["id"])
        passages.append(entry['text'])
        
    print(f"Loaded {len(passage_ids)} passages.")
    
    collection = Collection(data=passages)
    
    if return_ids:
        return collection, passage_ids
    
    return collection


def parse_pairs(filename):
    if "train" in filename:
        split = "train"
    elif "val" in filename or "test" in filename:
        split = "val"
    
    with open(filename, "r") as f:
        lines = f.readlines()
    
    data = []
    for line in lines:
        entry = json.loads(line.strip())
        question_id = int(entry["question_id"])
        image_id = entry["image_id"]
        question = entry["question"]
        # answers = entry["answers"]
        pos_passage = entry["pos_passage"]["passage"]
        # neg_passage = entry["neg_passage"]["passage"]
        
        image_path = imageid2path(image_id, split=split)
        
        data.append({
            "question": question, "document": pos_passage,
            "image": image_path, "q_id": question_id
        })
    
    return data


class OKVQARetrievalDataset(BaseRetrievalDataset):
    def __init__(self, data_path, img_dir, query_tokenizer, doc_tokenizer, img_processor, img_cached=False):
        super().__init__(img_dir, img_processor, query_tokenizer, doc_tokenizer, img_cached)
        
        samples = parse_pairs(data_path)
        
        # Caching
        if os.path.exists(data_path[:-3]+"pt"):
            print("Caching..")
            self.data = torch.load(data_path[:-3]+"pt")
        else:
            for sample in tqdm(samples):
                self.process_sample(sample)
            del samples
            
            torch.save(self.data, data_path[:-3]+"pt")
    
    def process_sample(self, sample):
        question = sample["question"]
        passage = sample["document"].strip()
        image_path = sample["image"]
        if self.img_cached:
            q_image_path = os.path.join(self.img_dir, image_path[:-3]+"npy")
        else:
            q_image_path = os.path.join(self.img_dir, image_path)
        
        self.data.append({
            "T_q_ids": self.tensorize_query(question), "T_d_ids": self.tensorize_doc(passage),
            "I_q": q_image_path, "I_d": [], # "A": answer, 
        })

class OKVQAGoogleSearchDataset(BaseRetrievalDataset):
    def __init__(self, data_path, img_dir, query_tokenizer, doc_tokenizer, img_processor, nways=64, caption=False, img_cached=False):
        super().__init__(img_dir, img_processor, query_tokenizer, doc_tokenizer, img_cached)
        
        self.nways = nways
        self.caption = caption
        samples = parse_pairs_for_gs(data_path, random_choice=True, caption=caption)

        # Caching
        # if os.path.exists(data_path[:-4]+"pt"):
        #     print("Caching..")
        #     self.data = torch.load(data_path[:-4]+"pt")
        # else:
        for sample in tqdm(samples):
            self.process_sample(sample)
        del samples
        
        # torch.save(self.data, data_path[:-4]+"pt")
    
    def process_sample(self, sample):
        question = sample["question"]        
        passage = sample["document"].strip()
        image_path = sample["image"]
        _, sp, img_id = image_path.split("_")
        image_path = imageid2path(img_id, sp.split("20")[0])
        if self.img_cached:
            q_image_path = os.path.join(self.img_dir, image_path[:-3]+"npy")
        else:
            q_image_path = os.path.join(self.img_dir, image_path)
        
        passage = self.tensorize_doc(passage)

        if self.nways > 1:
            # if not "negs" in sample or len(sample["negs"]) < self.nways-1:
            #     comp = random.choice(self.data)
            #     if not "negs" in sample:
            #         hard_negs = comp["negs"]
            #     else:
            #         hard_negs = sample["negs"] + random.sample(comp["negs"], self.nways-1-len(sample["negs"]))
            # else:
            if "negs" in sample:
                hard_negs = sample["negs"]
            else:
                print(sample)

        self.data.append({
            "T_q_ids": self.tensorize_query(question), "T_d_ids": passage,
            "I_q": q_image_path, "I_d": [] # "A": answer, 
        })

        if self.nways > 1:
            self.data[-1]["negs"] = hard_negs
        

class DynamicEval():
    def __init__(self, ann_file, ques_file, passage_id_to_line_id_file, all_blocks_file):
        
        with open(passage_id_to_line_id_file) as fin:
            self.passage_id_to_line_id = json.load(fin)
            
        self.vqa = VQA(ann_file, ques_file)
        self.all_blocks_file = all_blocks_file
            
    
    def get_answers(self, question_id):
        ann = self.vqa.loadQA(question_id)
        qa = self.vqa.returnQA(ann)[0]
        answers = set(answer.lower() for answer in qa['answers'].values() if answer)
        return answers
    
    
    def get_passage(self, passage_id):
        passage_line = linecache.getline(
            self.all_blocks_file, self.passage_id_to_line_id[passage_id])
        passage_dict = json.loads(passage_line)
        passage = passage_dict['text']
        assert passage_id == passage_dict['id']

        return passage
    
    
    def has_answers(self, answers, passage):
        passage = passage.lower()
        for answer in answers:
            answer = answer.lower()
            # "\b" matches word boundaries.
            # answer_starts = [match.start() for match in re.finditer(
            #     r'\b{}\b'.format(answer.lower()), passage)]
            if re.search(r'\b{}\b'.format(answer), passage):
                return True
        return False
    
    
    def gen_qrels(self, question_ids, I, retrieved_id_to_passage_id):
        qrels = defaultdict(dict)
        for question_id, retrieved_ids in tqdm(zip(question_ids, I), total=len(question_ids)):
            if question_id not in qrels:
                qrels[str(question_id)] = {'placeholder': 0}

            for retrieved_id in retrieved_ids:
                passage_id = retrieved_id_to_passage_id[retrieved_id]
                answers = self.get_answers(int(question_id))
                passage = self.get_passage(passage_id)

                if self.has_answers(answers, passage):
                    qrels[str(question_id)][passage_id] = 1

        return qrels