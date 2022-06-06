python run_infer.py \
--gpu=0 \
--nr_types=5 \
--type_info_path=type_info.json \
--batch_size=32 \
--model_mode='original' \
--model_path=./dataset/checkpoints/weight_decay_0.1/00/net_epoch=50.tar \
--nr_inference_workers=8 \
--nr_post_proc_workers=16 \
tile \
--input_dir=./dataset/CoNSeP/Test/Images/ \
--output_dir=./dataset/sample_tiles/pred/Res50_weight_decay \
--mem_usage=0.1 \
--draw_dot \
--save_qupath 
