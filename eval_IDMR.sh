#!/bin/bash

MODEL_NAME=lbw18601752667/IDMR-8B

DATASET=IDMR_test_coco  # "IDMR_test_coco" "IDMR_test_objects365" "IDMR_test_openimages" "IDMR_test_kitchen_instance" "IDMR_test_kitchen_location" "IDMR_test_lasot_instance" "IDMR_test_lasot_location"

IMAGE_DIR=./data/IDMR/test/images/

echo "Evaluating dataset: ${DATASET}"

DATASET_NAME=./data/IDMR/test/parquet/$DATASET
ENCODE_OUTPUT_PATH=./outputs/$DATASET/IDMR-8B

python eval.py --model_name $MODEL_NAME --model_backbone internvl_2_5\
  --encode_output_path $ENCODE_OUTPUT_PATH \
  --num_crops 4 --max_len 1024 \
  --pooling last --normalize True \
  --dataset_name $DATASET_NAME \
  --subset_name default \
  --dataset_split test --per_device_eval_batch_size 32 \
  --image_dir $IMAGE_DIR
