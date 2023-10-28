#!/bin/bash

source ./config.sh
MODELNAME=rcmmff_idmae_vit_base_patch16_dec512d8b
SAVEPATH=../checkpoints/${MODELNAME}/
mkdir -p ../${SAVEPATH}
cp ./pretrain.sh ../${SAVEPATH}

cd ..

export GOPEN_BUFFER=8192000

export PYTHONPATH=$(pwd):$PYTHONPATH

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 \
torchrun --nproc_per_node=8 --master_port 29501 main_pretrain.py \
--train_batch_size 108 72 56 40 32 24 \
--random_crop --crop_size 160 192 224 256 288 320 \
--model ${MODELNAME} --img_mask noise --feature_mask noise \
--warmup_epochs 5 --warmup-lr 1e-8 \
--wds_data_urls ${WDSIMAGENET1KTRAINURLS} --wds_data_total ${WDSIMAGENET1KTRAINTOTAL} \
--num_workers 4 --use_conv --chunks 4 --mask_chunks 4 \
--output_dir ${SAVEPATH} \
--start_epoch 0 --epochs 100 --mul_epochs 1 \
--blr 1e-4 --min_lr 1e-8


