# 🦮 Ret-XKnow: End-to-End **Ret**riever to e**X**pand visual **Know**ledge

Ret-XKnow endows a text retriever with the understanding of multimodal queries in a context of efficient information retrieval.

## Settings

1. Download the [pre-trained ColBERTv2 checkpoint](https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/colbertv2.0.tar.gz) to ckpts and unzip the downloaded file.

    We employ [ColBERTv2](https://github.com/stanford-futuredata/ColBERT) as a text retriever baseline.

2. Install python packages.

    ~~~bash
    pip3 install -r requirements.txt
    ~~~

3. Download downstream task datasets.

    We use two retrieval datasets curated from OK-VQA. Additionally, we conducted experiments for the Infoseek datset. You can download the dataset in the following links:

    - [OK-VQA (Wiki-11M)](https://github.com/prdwb/okvqa-release)
    - [OK-VQA (Google Search)](https://github.com/LinWeizheDragon/Retrieval-Augmented-Visual-Question-Answering?tab=readme-ov-file#download-datasets)
        > In this dataset, questions in the annotation files include captions for images. Thus, we edit the questions to remove captions. See `dataset/vqa_ret.py` for details.
    - [InfoSeek (Wiki-6M)](https://github.com/open-vision-language/infoseek)

## Visual Dialogue-to-Retrieval (ViD2R) Dataset Construction

1. Download instruction data and image datasets from the following pages: 

    - [Visual instruction dataset](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#visual-instruction-tuning) (Here, download images with the dialogue dataset)

    - [LVIS-Instruct4V](https://huggingface.co/datasets/X2FD/LVIS-Instruct4V)

2. Pre-processing and neural filtering using a text retriever:
    
    You can skip the neural filtering step by modifying the code if you want to build the dataset fast.

    ~~~bash
    python3 -m runs.neural_filtering --data_path1 [path to data1] --data_path2 [path to data2] --colbert_ckpt [directory with colbert checkpoint] --save_path [path to save]
    ~~~

3. Converting responses to passages:

    We require a knowledge base (KB) and a text retriever to convert dialogues to retrieval tasks. We adopt 6M Wikipedia passages as the KB. You can download the passages in this [link](http://storage.googleapis.com/gresearch/open-vision-language/Wiki6M_ver_1_0.jsonl.gz).

    ~~~bash
    python3 -m runs.convert_tasks --data_path [path to pre-processed data] --colbert_ckpt [directory with colbert checkpoint] --db_pool [path to KB] --save_path data/vid2r/ViD2R.json
    ~~~

## Training Ret-XKnow

First, set configure files!

**Pre-training Ret-XKnow on the ViD2R datset**

~~~bash
export WANDB_API_KEY=$Your_WANDB_KEY
CONFIG_PATH=cfgs/xknow_train_vid2r.yaml

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NPROC=8

# Caching visual embeddings
python3 -m runs.run_visual_embedder --model_name openai/clip-vit-base-patch32 --data_path data/vid2r/ViD2R.json --batch_size 512 --image_dir data/vid2r/images

# Run training command
python3 -m torch.distributed.run --nproc_per_node=$NPROC train_retrieval.py --config_path "$CONFIG_PATH"
~~~

or 

After modifying the shell file `scripts/pretrain_xknow_inbatch.sh` to your path, execute the following command:

~~~bash
bash scripts/pretrain_xknow_inbatch.sh
~~~

This shell file evaluates zero-shot performance after training. If indexing has already been done, comment out the execution of run_indexer.

**Fine-tuning Ret-XKnow on the downstream task**

