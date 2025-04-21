MODEL_NAME=OpenGVLab/InternVL2_5-8B

DATA_DIR=./data/IDMR/train/parquet
IMAGE_DIR=./data/IDMR/train/images
OUTPUT_DIR=./ckpt/IDMR-8B

WANDB_API_KEY=YOUR_WANDB_API_KEY
wandb login --relogin $WANDB_API_KEY

torchrun --nproc_per_node=8 --master_port=22459 --max_restarts=0 train.py \
 --model_name $MODEL_NAME --model_backbone internvl_2_5 --bf16 --pooling last \
 --dataset_name $DATA_DIR  \
 --lora_target_modules qkv,wqkv,wo,w1,w2,w3 \
 --subset_name MMEB_train IDMR_train_coco IDMR_train_objects365 IDMR_train_openimages \
 --image_dir $IMAGE_DIR \
 --max_len 1024 --output_dir $OUTPUT_DIR --logging_steps 20 \
 --lr_scheduler_type linear --learning_rate 2e-5 --num_train_epochs 1 \
 --warmup_steps 120 --save_steps 100 --normalize True \
 --temperature 0.02 --per_device_train_batch_size 64 \
 --lora --lora_r 8 \
 --grad_cache True --gc_q_chunk_size 8 --gc_p_chunk_size 8 --wandb True\
