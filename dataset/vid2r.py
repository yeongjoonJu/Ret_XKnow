import json, os
import numpy as np
from tqdm import tqdm
import torch
from dataset.base import BaseRetrievalDataset
# from retrievers.vilt.transforms.pixelbert import pixelbert_transform

class ViD2RDataset(BaseRetrievalDataset):
    def __init__(self, data_path, img_dir, query_tokenizer, doc_tokenizer, img_processor, img_cached=False):
        super().__init__(img_dir, img_processor, query_tokenizer, doc_tokenizer, img_cached)
        
        # Caching
        if os.path.exists(data_path[:-4]+"pt"):
            self.data = torch.load(data_path[:-4]+"pt")
        else:
            with open(data_path, "r") as f:
                samples = json.load(f)
            for sample in tqdm(samples, desc="Processing samples"):
                self.process_sample(sample)
            torch.save(self.data, data_path[:-4]+"pt")
        
            del samples

    def process_sample(self, sample):
        question = sample["question"]
        passage = sample["document"].strip()
        image_path = sample["image"]
        if self.img_cached:
            q_image_path = os.path.join(self.img_dir, image_path[:-3]+"npy")
        else:
            q_image_path = os.path.join(self.img_dir, image_path)

        T_q_ids = self.tensorize_query(question)
        
        self.data.append({
            "T_q_ids": T_q_ids, "T_d_ids": self.tensorize_doc(passage),
            "I_q": q_image_path, "I_d": [], # "A": answer, 
        })
        
"""
class ReVizViD2RDataset(BaseRetrievalDataset):
    def __init__(self, data_path, img_dir, tokenizer, image_size=384, img_cached=False):
        super().__init__(
            img_dir,
            query_tokenizer=tokenizer,
            doc_tokenizer=tokenizer,
            img_processor=pixelbert_transform(size=image_size),
            img_cached=img_cached)
        
        with open(data_path, "r") as f:
            samples = json.load(f)
        for sample in tqdm(samples, desc="Processing samples"):
            self.process_sample(sample)
        
        del samples
        
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
    
    def tensorize_query(self, text):
        return self.query_tokenizer([text],
                padding="longest", truncation="longest_first",
                return_tensors="pt", max_length=64).input_ids[0]
    
    def tensorize_doc(self, text):
        return self.doc_tokenizer([text],
                padding="longest", truncation="longest_first",
                return_tensors="pt", max_length=384).input_ids[0]
"""