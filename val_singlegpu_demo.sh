#!/bin/bash

# Set CUDA device
export CUDA_VISIBLE_DEVICES="0"

# Define variables
arch="vit_b"  # Change this value as needed
finetune_type="lora"
dataset_name="DigitalPathology"  # Assuming you set this if it's dynamic
targets='combine_all'


# Construct the checkpoint directory argument
dir_checkpoint="2D-SAM_${arch}_encoderdecoder_${finetune_type}_${dataset_name}_noprompt"

img_folder="./datasets"
val_img_list="${img_folder}/${dataset_name}/valid_DPdata.csv"

# Run the Python script
python val_finetune_noprompt.py \
    -if_warmup True \
    -finetune_type "$finetune_type" \
    -arch "$arch" \
    -if_mask_decoder_adapter True \
    -img_folder "$img_folder" \
    -mask_folder "$img_folder" \
    -dataset_name "$dataset_name" \
    -dir_checkpoint "$dir_checkpoint"\
    -val_img_list "$val_img_list" 