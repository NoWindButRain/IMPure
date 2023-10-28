#!/bin/bash

source ./config.sh

cd ..

# RESNET101V2NAME INCEPTIONV3NAME INCEPTIONRESNETV2NAME
TARGETNAME=${RESNET101V2NAME}
TARGETSIZE=224
ROOTPATH=${ADVPURIROOT}/attack_fix_recon
# RESNET101V2VALJSON INCEPTIONV3VALJSON INCEPTIONRESNETV2VALJSON
FILTERDATAPATH=${FILTERDATAROOT}/${RESNET101V2VALJSON}
ATTACKEPS=16 # 4 16
GETATTACKNAMES ${TARGETNAME} ${ATTACKEPS} 4

python3 data_process/data_val.py --device 'cuda:1' \
--model ${TARGETNAME} --input_size ${TARGETSIZE} \
--root_path ${IMAGENET1KROOT} \
--meta_path ${IMAGENET1KROOT} \
--data_split val --filter_data ${FILTERDATAPATH}


for ATTACKNAME in ${ATTACKNAMES[*]}; do
  if [ -d ${ROOTPATH}/${ATTACKNAME} ]; then
  echo $ATTACKNAME
    python3 data_process/data_val.py --device 'cuda:0' \
--model ${TARGETNAME} --input_size ${TARGETSIZE} \
--adv_attack --root_path ${ROOTPATH}/${ATTACKNAME} \
--meta_path ${IMAGENET1KROOT} \
--data_split val --filter_data ${FILTERDATAPATH}

  fi
done


