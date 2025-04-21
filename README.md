# IDMR

The official repo for [IDMR: Towards Instance-Driven Precise Visual Correspondence in Multimodal Retrieval](https://arxiv.org/pdf/2504.00954). 


<a target="_blank" href="https://arxiv.org/pdf/2504.00954">
<img style="height:22pt" src="https://img.shields.io/badge/-Paper-black?style=flat&logo=arxiv"></a>
<a target="_blank" href="https://github.com/BwLiu01/IDMR">
<img style="height:22pt" src="https://img.shields.io/badge/-Code-green?style=flat&logo=github"></a>
<a target="_blank" href="https://huggingface.co/lbw18601752667/IDMR-8B">
<img style="height:22pt" src="https://img.shields.io/badge/-🤗%20Models-red?style=flat"></a>
<a target="_blank" href="https://huggingface.co/spaces/lbw18601752667/IDMR-demo">
<img style="height:22pt" src="https://img.shields.io/badge/-🤗%20Demo-red?style=flat"></a>


## Model
 - [IDMR-8B](https://huggingface.co/lbw18601752667/IDMR-8B)
 - [IDMR-26B](https://huggingface.co/lbw18601752667/IDMR-26B)
 - More to come!

## Installation
```bash
git clone https://github.com/BwLiu01/IDMR.git
cd IDMR
pip install -r requirements.txt
```

## Inference & Examples
- Inference examples with Gradio: [IDMR-Demo](https://huggingface.co/spaces/lbw18601752667/IDMR-demo)
- Inference locally:
```bash
python inference.py
```

## Data
Comming soon

## Training
Run the following script to train IDMR.
```bash
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
```


## Acknowledgement
We have adapted code from [VLM2Vec](https://github.com/TIGER-AI-Lab/VLM2Vec), a comprehensive implementation of transforming MLLMs to embedding models.


