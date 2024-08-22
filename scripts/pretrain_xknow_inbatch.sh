# Update config
export WANDB_API_KEY=9a6625212c3e9beff56070d25b7ac5667d550cb3

CONFIG_PATH=cfgs/xknow_train_vid2r.yaml

# Set CUDA devices and PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 # <--- Change this to the CUDA devices you want to us
NPROC=8

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "CONFIG_PATH: $CONFIG_PATH"

# Caching visual embeddings
python3 -m runs.run_visual_embedder --model_name openai/clip-vit-base-patch32 --data_path data/vid2r/ViD2R.json --batch_size 512 --image_dir data/vid2r/images

# Run training command
python3 -m torch.distributed.run --nproc_per_node=$NPROC train_retrieval.py --config_path "$CONFIG_PATH"

CHECKPOINT=ckpts/xknow/vid2r/base/xknow_epoch_9.pth
echo $CHECKPOINT

# Eval okvqa-gs
PASSAGES=data/okvqa/RAVQA_v2_data/okvqa/pre-extracted_features/passages/okvqa_full_clean_corpus.csv
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m runs.run_indexer --exp_name okvqa_gs --n_bits 2 --dataset_name okvqa_gs --all_blocks_file $PASSAGES
CUDA_VISIBLE_DEVICES=0 python3 -m runs.evaluate_retrieval \
    --dataset_name okvqa_gs \
    --index_name okvqa_gs.nbits=2 \
    --save_path results/eval_okvqa_gs_xknow_base_zero.json \
    --all_blocks_file $PASSAGES \
    --anno_file data/okvqa/RAVQA_v2_data/okvqa/pre-extracted_features/passages/retriever_test.json \
    --xknow_ckpt $CHECKPOINT \
    --image_dir data/vid2r/images/coco/val2014

# Eval okvqa-wiki
PASSAGES=data/okvqa/all_blocks.txt
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m runs.run_indexer --exp_name okvqa --n_bits 2 --dataset_name okvqa --all_blocks_file $PASSAGES
CUDA_VISIBLE_DEVICES=0 python3 -m runs.evaluate_retrieval \
    --dataset_name okvqa \
    --index_name okvqa.nbits=2 \
    --save_path results/eval_okvqa_wiki_xknow_base_zero.json \
    --image_dir data/vid2r/images \
    --xknow_ckpt $CHECKPOINT \
    --all_blocks_file $PASSAGES \
    --anno_file data/okvqa/test2014_pairs_cap_combine_sum.txt

# Eval ReMuQ
PASSAGES=data/remuq/open_corpus_195837.csv
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m runs.run_indexer --exp_name remuq --n_bits 2 --dataset_name remuq --all_blocks_file $PASSAGES
CUDA_VISIBLE_DEVICES=0 python3 -m runs.evaluate_retrieval \
    --dataset_name remuq \
    --index_name remuq.nbits=2 \
    --save_path results/eval_remuq_xknow_base_zero.json \
    --all_blocks_file $PASSAGES \
    --anno_file data/remuq/test.json \
    --xknow_ckpt $CHECKPOINT \
    --image_dir data/remuq

# Eval A-OKVQA
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m runs.run_indexer --exp_name aokvqa --n_bits 2 --dataset_name aokvqa --all_blocks_file data/aokvqa/aokvqa_v1p0_val.json
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m runs.evaluate_retrieval \
    --dataset_name aokvqa \
    --index_name aokvqa.nbits=2 \
    --save_path results/eval_aokvqa_xknow_zero.json \
    --anno_file data/aokvqa/aokvqa_v1p0_val.json \
    --image_dir data/vid2r/images/coco/val2017 \
    --xknow_ckpt $CHECKPOINT