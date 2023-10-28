#!/bin/bash

source ./config.sh

GETATTACKNAMES ${RESNET101V2NAME} 16 4
ADVNAME4=(${ATTACKNAMES[*]})
GETATTACKNAMES ${INCEPTIONV3NAME} 16 4
ADVNAME7=(${ATTACKNAMES[*]})
GETATTACKNAMES ${INCEPTIONRESNETV2NAME} 16 4
ADVNAME8=(${ATTACKNAMES[*]})
ADVNAMES=(${ADVNAME4[*]} ${ADVNAME7[*]} ${ADVNAME8[*]})
ADVVALNAMES=(${ADVNAME7[*]})


ADVWDSIMAGENET1KPATH="/data/object_class/adverpuri/attack_fix_wds/"
ADVWDSSPLIT="train"
GETADVWDSURLS ${ADVNAMES[*]}
ADVWDSURLS1=(${ADVWDSURLS[*]})
ADVWDSTRAINURLS=(${ADVWDSURLS1[*]})
echo ${#ADVWDSTRAINURLS[*]}
ADVWDSTRAINTOTAL=300000

ADVWDSSPLIT="val"
GETADVWDSURLS ${ADVVALNAMES[*]}
ADVWDSURLS1=(${ADVWDSURLS[*]})
ADVWDSVALURLS=(${ADVWDSURLS1[*]})
echo ${#ADVWDSVALURLS[*]}
ADVWDSVALTOTAL=50000

MODELNAME=rcmmff_idmae_vit_base_patch16_dec512d8b
SAVEPATH=../checkpoints/${MODELNAME}_${VGG19BNV1NAME}/
mkdir -p ../${SAVEPATH}
cp ./advpuri_mipure_train.sh ../${SAVEPATH}

cd ..

export GOPEN_BUFFER=8192000

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 OMP_NUM_THREADS=1 \
torchrun --nproc_per_node=6 --master_port 29501 main_finetune.py \
--model ${MODELNAME} --img_mask noise --feature_mask noise \
--warmup_epochs 5 --warmup-lr 1e-8 \
--adv_wds_data_urls ${ADVWDSTRAINURLS[*]} \
--wds_data_totals 100000 100000 100000 \
--wds_data_url_nums 1000 1000 1000 \
--train_batch_sizes 56 32 32 \
--random_crop --crop_size 192 256 256 \
--adv_wds_val_data_urls ${ADVWDSVALURLS[*]} --wds_val_data_total ${ADVWDSVALTOTAL} \
--val_batch_size 64 \
--val_modelname ${INCEPTIONV3NAME} \
--num_workers 4 --use_conv --chunks 4 --mask_chunks 4 \
--hgds "[{\"name\": \"${VGG19BNV1NAME}\", \
\"out_feature\": [\"features.38\", \"features.51\", \"classifier.1\"], \
\"out_hyper\": [30, 10, 10]}]" \
--output_dir ${SAVEPATH} --save_inter 5 \
--start_epoch 0 --epochs 100 --blr 1e-4 --min_lr 1e-8 \
--resume <pretrain weight>

