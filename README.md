# 🦮 Ret-XKnow: End-to-End Multi-modal Retriever

**End-to-End Multi-modal Retriever for Conveying Explicit Visual and Textual Knowledge**

## Settings

We employ [ColBERTv2](https://github.com/stanford-futuredata/ColBERT) as a text retriever baseline.

1. Download the [pre-trained ColBERTv2 checkpoint](https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/colbertv2.0.tar.gz).


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
    python3 -m runs.convert_tasks --data_path [path to pre-processed data] --colbert_ckpt [directory with colbert checkpoint] --db_pool [path to KB] --save_path data/vid2r/vid2r.json
    ~~~