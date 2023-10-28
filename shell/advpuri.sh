#!/bin/bash

source ./config.sh

cd ..

ROOTPATH=${ADVPURIROOT}/attack_fix
SAVEPATH=${ADVPURIROOT}/attack_orisize
CHECKPOINTPATH=${CHECKPOINTROOT}/<weight path>
TARGETNAME=${RESNET101V2NAME}
ATTACKEPS=16 # 4 16
GETATTACKNAMES ${TARGETNAME} ${ATTACKEPS} 4

python3 data_process/main_advpur.py --batch_size 64 --device "cuda:1" \
--model rcmmff_idmae_vit_base_patch16_dec512d8b \
--data_split val --img_mask noise --feature_mask noise  \
--meta_path ${IMAGENET1KROOT} \
--filter_data ${FILTERDATAROOT}/${RESNET101V2TESTJSON} \
--root_path ${IMAGENET1KROOT} \
--save_path ${SAVEPATH}/ILSVRC2012 \
--resume ${CHECKPOINTPATH} \
--target_model ${TARGETNAME} --val_model \
--val_preprocess
#--save_img

for ATTACKNAME in ${ATTACKNAMES[*]}; do
  if [ -d ${ROOTPATH}/${ATTACKNAME} ]; then
  echo $ATTACKNAME
  python3 data_process/main_advpur.py --batch_size 64 --device "cuda:0" \
--model rcmmff_idmae_vit_base_patch16_dec512d8b \
--adv_attack --img_mask noise --feature_mask noise --data_split val \
--meta_path ${IMAGENET1KROOT} \
--filter_data ${FILTERDATAROOT}/${RESNET101V2TESTJSON} \
--root_path ${ROOTPATH}/${ATTACKNAME} \
--save_path ${SAVEPATH}/${ATTACKNAME} \
--resume ${CHECKPOINTPATH} \
--target_model ${TARGETNAME} --val_model \
--val_preprocess \
#--save_img

  fi
done






