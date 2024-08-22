CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m runs.neural_filtering \
    --data_paths data/vid2r/llava_v1_5_mix665k.json data/vid2r/lvis_instruct4v_220k.json \
    --save_path data/vid2r/ViD2R_filtered_wo_connect.json

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m runs.convert_tasks \
    --data_path data/vid2r/ViD2R_filtered_wo_connect.json \
    --save_path data/vid2r/ViD2R.json