import json, os
import torch
from io import BytesIO
import base64
import random
from PIL import Image, ImageFile
import numpy as np
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader, DistributedSampler
from model_utils.dist import ContiguousDistributedSampler

import retrievers.utils as utils
# from retrievers.vilt.transforms.pixelbert import pixelbert_transform

MASK_TOKEN_ID = 103
PAD_TOKEN_ID = 0

def convert_list2dict(data, set_key="id"):
    converted = {}
    for d in data:
        converted[d[set_key]] = {k: v for k, v in d.items() if k!=set_key}
    
    return converted

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            json_obj = json.loads(line)
            data.append(json_obj)
    return data

class BaseRetrievalDataset(Dataset):
    def __init__(self, img_dir, img_processor, query_tokenizer, doc_tokenizer, img_cached=False,):
        self.data = []
        self.img_dir = img_dir
        self.img_processor = img_processor
        self.query_tokenizer = query_tokenizer
        self.doc_tokenizer = doc_tokenizer
        self.img_cached = img_cached
    
    def process_sample(self, sample):
        """
        T_q: textual query, A: answer, T_d: positive document, I_d: positive reference image,
        T_d_neg: negative documents, T_d_neg: negative reference images
        """
        pass
    
    def __getitem__(self, index):
        sample = {
            "T_q_ids": None, "T_q_masks": None, "I_q": None, "I_q_emb": None, "I_q_masks": None,
            "T_d_ids": None, "T_d_masks": None, "I_d": None, "I_d_emb": None, "I_d_masks": None,
        }
        # for k, v in self.data[index].items():
        #     sample[k] = v
            
        # Tokenize and encode
        sample["T_q_ids"] = self.data[index]["T_q_ids"]
        sample["T_d_ids"] = self.data[index]["T_d_ids"]
        if "negs" in self.data[index]:
            negs = random.sample(self.data[index]["negs"], self.nways-1)
            sample["T_d_ids"] = [sample["T_d_ids"]]
            sample["T_d_ids"].extend([self.tensorize_doc(neg) for neg in negs])

        # Load image
        if self.data[index]["I_d"]:
            I_d_path = self.data[index]["I_d"]
            if self.img_cached:
                sample["I_d_emb"] = torch.tensor(np.load(I_d_path))
            else:
                if self.data[index]["I_d"][-3:]=="jpg":
                    I_d = Image.open(I_d_path).convert("RGB")
                else:
                    with open(I_d_path, "r") as fp:
                        img_base64 = fp.readline().strip()
                        I_d = Image.open(BytesIO(base64.b64decode(img_base64)))
                
                pixels = self.img_processor(I_d)
                if type(pixels) is dict and "pixel_values" in pixels:
                    pixels = pixels["pixel_values"][0]
                sample["I_d"] = torch.tensor(pixels)
            sample["I_d_masks"] = torch.tensor([1.])
        else:
            # sample["I_d"] = torch.zeros((3, 224, 224))
            sample["I_d_masks"] = torch.tensor([0.])
            # sample["I_d"] = None
            # sample["I_d_masks"] = None
            
        if self.data[index]["I_q"]:
            I_q_path = self.data[index]["I_q"]
            if self.img_cached:
                sample["I_q_emb"] = torch.tensor(np.load(I_q_path))
            else:
                I_q = Image.open(I_q_path).convert("RGB")
                pixels = self.img_processor(I_q)
                if "pixel_values" in pixels:
                    pixels = pixels["pixel_values"][0]
                sample["I_q"] = torch.tensor(pixels)
            sample["I_q_masks"] = torch.tensor([1.])
        else:
            # sample["I_q"] = torch.zeros((3, 224, 224))
            sample["I_q_masks"] = torch.tensor([0.])
        
        return sample
    
    def __len__(self):
        return len(self.data)
    
    def tensorize_query(self, text):
        max_len = self.query_tokenizer.query_maxlen
        ids = self.query_tokenizer.tok(['. ' + text], padding="max_length", truncation=True,
                                   return_tensors="pt", max_length=max_len).input_ids
        
        # postprocess for the [Q] marker and the [MASK] augmentation
        ids[:, 1] = self.query_tokenizer.Q_marker_token_id
        ids[ids == self.query_tokenizer.pad_token_id] = self.query_tokenizer.mask_token_id
        
        return ids[0]
    
    def tensorize_doc(self, text):
        max_len = self.doc_tokenizer.doc_maxlen
        ids = self.doc_tokenizer.tok(['. ' + text], padding="longest", truncation="longest_first",
                                     return_tensors="pt", max_length=max_len).input_ids
        
        # postprocess for the [D] marker
        ids[:, 1] = self.doc_tokenizer.D_marker_token_id
        
        return ids[0]
    
    def tensorize_docs(self, docs):
        max_len = self.doc_tokenizer.doc_maxlen
        ids = self.doc_tokenizer.tok(['. ' + doc for doc in docs], padding="longest", truncation="longest_first",
                                     return_tensors="pt", max_length=max_len).input_ids
        
        # postprocess for the [D] marker
        ids[:, 1] = self.doc_tokenizer.D_marker_token_id
        
        return ids
    

def collate_fn(batch):
    T_q_ids = torch.nn.utils.rnn.pad_sequence(
        [b["T_q_ids"] for b in batch], batch_first=True, padding_value=MASK_TOKEN_ID)
    # T_q_masks = T_q_ids.ne(MASK_TOKEN_ID)
    T_q_masks = torch.logical_or(T_q_ids.ne(MASK_TOKEN_ID), T_q_ids.ne(PAD_TOKEN_ID)) # For ReViz
    
    if type(batch[0]["T_d_ids"]) is list:
        T_d_ids = []
        for b in batch:
            T_d_ids.extend(b["T_d_ids"])
    else:
        T_d_ids = [b["T_d_ids"] for b in batch]
        
    T_d_ids = torch.nn.utils.rnn.pad_sequence(
        T_d_ids, batch_first=True, padding_value=PAD_TOKEN_ID)
    T_d_masks = T_d_ids.ne(PAD_TOKEN_ID)
    
    I_q_masks = None
    I_d_masks = None

    if batch[0]["I_q_masks"] is not None:
        I_q_masks = torch.stack([b["I_q_masks"] for b in batch], dim=0)
        
    if batch[0]["I_d_masks"] is not None:
        I_d_masks = torch.stack([b["I_d_masks"] for b in batch], dim=0)
    
    # Optional: Negative samples 처리
    # T_d_neg_ids, T_d_neg_masks = doc_tokenizer.tensorize([b["T_d_neg"] for b in batch])
    # I_d_neg = torch.stack([b["I_d_neg"] for b in batch], dim=0)
        
    I_q = torch.stack([b["I_q"] for b in batch], dim=0) if batch[0]["I_q"] is not None else None
    I_d = torch.stack([b["I_d"] for b in batch], dim=0) if batch[0]["I_d"] is not None else None
    I_q_emb = torch.stack([b["I_q_emb"] for b in batch], dim=0) if batch[0]["I_q_emb"] is not None else None
    I_d_emb = torch.stack([b["I_d_emb"] for b in batch], dim=0) if batch[0]["I_d_emb"] is not None else None
    
    return {
        "T_q_ids": T_q_ids, "T_q_masks": T_q_masks, "I_q": I_q, "I_q_emb": I_q_emb, "I_q_masks": I_q_masks,
        "T_d_ids": T_d_ids, "T_d_masks": T_d_masks, "I_d": I_d, "I_d_emb": I_d_emb, "I_d_masks": I_d_masks
    }
    

def get_dataloader(dataset, batch_size=1, shuffle=False, num_workers=4):
    """
    T_q: textual query, A: answer, T_d: positive document, I_d: positive reference image,
    T_d_neg: negative documents, T_d_neg: negative reference images
    """
    
    sampler = DistributedSampler(
        dataset=dataset,
        num_replicas=utils.get_world_size(),
        rank=utils.get_rank(),
        shuffle=shuffle,
    )

    dataloader = DataLoader(
        dataset, 
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=True,
        shuffle=False, # Note: since we use sampler, shuffle should be False
        batch_size=batch_size,
        drop_last=shuffle,
        num_workers=num_workers
    )

    return dataloader


def collate_fn_for_emb(batch, doc_tokenizer):
    T_d_ids, T_d_masks = doc_tokenizer.tensorize([b["T_d"] for b in batch])
    I_d = torch.stack([b["I_d"] for b in batch], dim=0)
    I_d_masks = torch.stack([b["I_d_masks"] for b in batch], dim=0)
    q_ids = [b["id"] for b in batch]

    return {
        "T_d_ids": T_d_ids, "T_d_masks": T_d_masks, "I_d": I_d, "I_d_masks": I_d_masks,
        "q_id": q_ids
    }

def get_dataloader_for_embedding(dataset, doc_tokenizer, batch_size=1):
    sampler = ContiguousDistributedSampler(
        dataset=dataset,
        num_replicas=utils.get_world_size(),
        rank=utils.get_rank()
    )
    
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        collate_fn=lambda batch: collate_fn_for_emb(batch, doc_tokenizer),
        pin_memory=True,
        shuffle=False,
        batch_size=batch_size,
        drop_last=False
    )
    
    return dataloader

        
class ImageDataset(Dataset):
    def __init__(self, img_list, img_dir, img_processor):
        img_list = [os.path.join(img_dir, path) for path in img_list]
        self.img_list = []
        num_removed = 0
        for i in range(len(img_list)):
            if not os.path.exists(img_list[i]):
                num_removed += 1
            else:
                self.img_list.append(img_list[i])
        
        print("The number of removed data:", num_removed)
        print("The number of existing data:", len(self.img_list))
        self.img_processor = img_processor
    
    def __getitem__(self, index):
        img_path = self.img_list[index]
        image = Image.open(img_path).convert("RGB")
        return torch.tensor(self.img_processor(image)["pixel_values"][0]), img_path
    
    def __len__(self):
        return len(self.img_list)
    
    
class DataPool:
    def __init__(self, anno_path, db_path, 
                 query_tokenizer, doc_tokenizer, img_processor,
                 image_dir=None, max_len=384, dataset_name="infoseek"):
        self.query_tokenizer = query_tokenizer
        self.doc_tokenizer = doc_tokenizer
        self.query_tokenizer.model_max_length = 64
        self.doc_tokenizer.model_max_length = max_len
        if dataset_name=="okvqa":
            from dataset.okvqa import get_collection, imageid2path
            
            if "train" in anno_path:
                split = "train"
            elif "val" in anno_path or "test" in anno_path:
                split = "val"
            with open(anno_path, "r") as f:
                lines = f.readlines()
            
            self.question_ids = []
            self.questions = []
            self.images = []
            for line in lines:
                entry = json.loads(line.strip())
                question_id = int(entry["question_id"])
                image_id = entry["image_id"]
                question = entry["question"]
                answers = entry["answers"]
                self.question_ids.append(question_id)
                self.questions.append(question)
                self.images.append(f"{image_dir}/{imageid2path(image_id, split=split)}")
            
        elif dataset_name in ["remuq", "okvqa_gs"]:
            from dataset.vqa_ret import get_collection
        elif dataset_name=="infoseek":
            from dataset.infoseek import get_collection
        
        doc_info = get_collection(db_path, return_ids=True)
        self.collection = doc_info[0]
        self.passage_ids = doc_info[1]
        if len(doc_info)==3:
            self.pid2passage = doc_info[2]
    
    def __len__(self):
        return len(self.collection)
            
    def __getitem__(self, index):
        question_id = self.question_ids[index]
        question = self.questions[index]
        image_path = self.images[index]
        passage_id = self.passage_ids[index]
        passage = self.collection[index]
        
        q_tokenized = self.query_tokenizer(question, padding="max_length", truncation=True)
        q_input_ids = q_tokenized["input_ids"]
        q_attention_mask = q_tokenized["attention_mask"]
        p_tokenized = self.doc_tokenizer(passage, padding="max_length", truncation=True)
        p_input_ids = p_tokenized["input_ids"]
        p_attention_mask = p_tokenized["attention_mask"]
        image = torch.tensor(self.img_processor(Image.open(image_path))["pixel_values"][0])
        
        return {"p_input_ids": p_input_ids, "p_attn_mask": p_attention_mask, "p_id": passage_id, \
                "q_input_ids": q_input_ids, "q_attn_mask": q_attention_mask, "q_id": question_id,
                "image": image}


def collate_fn_for_db(batch):
    p_input_ids = torch.stack([b["p_input_ids"] for b in batch], dim=0)
    p_attention_mask = torch.stack([b["p_attn_mask"] for b in batch], dim=0)
    p_ids = [b["p_id"] for b in batch]
    
    q_input_ids = torch.stack([b["q_input_ids"] for b in batch], dim=0)
    q_attention_mask = torch.stack([b["q_attn_mask"] for b in batch], dim=0)
    q_ids = [b["q_id"] for b in batch]
    
    image = torch.stack([b["image"] for b in batch], dim=0) if batch[0]["image"] is not None else None

    return {"p_id": p_ids, "p_input_ids": p_input_ids, "p_attention_mask": p_attention_mask,
            "q_id": q_ids, "q_input_ids": q_input_ids, "q_attention_mask": q_attention_mask,
            "image": image}