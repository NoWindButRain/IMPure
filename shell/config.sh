#!/bin/bash


FILTERDATAROOT='/data/results'

IMAGENET1KROOT='/data/object_class/ILSVRC2012'

WDSIMAGENET1KROOT="/data/object_class/ILSVRC2012_wds/"

ADVWDSIMAGENET1KROOT="/data/object_class/adverpuri/attack_fix_wds/"

ADVPURIROOT='/data/object_class/adverpuri'

CHECKPOINTROOT='/data/checkpoints'

VGG16BNV1NAME="vgg16bn_v1"
VGG19BNV1NAME="vgg19bn_v1"

RESNET101V2NAME="resnet101_v2"
RESNET101V2REAINJSON="ImageNet_train_resnet101_v2_10.json"
RESNET101V2VALJSON="ImageNet_val_resnet101_v2_5.json"
RESNET101V2TESTJSON="ImageNet_test_resnet101_v2_5.json"

INCEPTIONV3NAME="inceptionv3_v1"
INCEPTIONV3TRAINJSON="ImageNet_train_inceptionv3_v1_10.json"
INCEPTIONV3VALJSON="ImageNet_val_inceptionv3_v1_5.json"
RESNET101V2TESTJSON="ImageNet_val_resnet101_v2_5.json"

INCEPTIONRESNETV2NAME="inceptionresnetv2_v1"
INCEPTIONRESNETV2TRAINJSON="ImageNet_train_inceptionresnetv2_v1_10.json"
INCEPTIONRESNETV2VALJSON="ImageNet_val_inceptionresnetv2_v1_5.json"
RESNET101V2TESTJSON="ImageNet_test_resnet101_v2_5.json"

function GETATTACKNAMES() {
    ATTACKNAMES=(${1}_raw ${1}_FGSM_${2} \
${1}_CW2_${2}_5_40_0.01 ${1}_DeepFool_${2}_50_0.02 \
${1}_PGD_${2}_10_${3} ${1}_MIFGSM_${2}_10_${3} \
${1}_DIFGSM_${2}_10_${3}_0 ${1}_DIFGSM_${2}_10_${3}_1 \
${1}_APGD_${2}_10_${3}_ce ${1}_APGD_${2}_10_${3}_dlr)
}


WDSIMAGENET1KTRAINURLS=$(ls -d ${WDSIMAGENET1KROOT}train/*)
WDSIMAGENET1KTRAINTOTAL=1281000
WDSIMAGENET1KTVALURLS=$(ls -d ${WDSIMAGENET1KROOT}val/*)
WDSIMAGENET1KTVALTOTAL=50000


ADVWDSSPLIT="train"
function GETADVWDSURLS() {
  ADVWDSURLS=()
  USEATTACKNAMES=($@)
  for EACHATTACKNAMES in ${USEATTACKNAMES[*]}
  do
    if [ ${EACHATTACKNAMES} ]; then
      ADDURLS=($(ls -d ${ADVWDSIMAGENET1KROOT}/${EACHATTACKNAMES}/${ADVWDSSPLIT}/*))
      ADVWDSURLS=(${ADVWDSURLS[*]} ${ADDURLS[*]})
    fi
  done
}


