#!/bin/bash

SEG_PRETRAINED_ROOT=$HOME/workspace/semantic-segmentation-pytorch

ALL_MODEL_NAMES=("ade20k-hrnetv2-c1 30"
                 "ade20k-mobilenetv2dilated-c1_deepsup 20"
                 "ade20k-resnet18dilated-c1_deepsup 20"
                 "ade20k-resnet18dilated-ppm_deepsup 20"
                 "ade20k-resnet50-upernet 30"
                 "ade20k-resnet50dilated-ppm_deepsup 20"
                 "ade20k-resnet101-upernet 50"
                 "ade20k-resnet101dilated-ppm_deepsup 25")

for i in "${!ALL_MODEL_NAMES[@]}"; do
    MODEL=${ALL_MODEL_NAMES[i]}

    set -- $MODEL
    MODEL_NAME=$1
    CKPT_NUM=$2

    MODEL_PATH=${SEG_PRETRAINED_ROOT}/ckpt/$MODEL_NAME
    ENCODER=$MODEL_NAME/encoder_epoch_${CKPT_NUM}.pth
    DECODER=$MODEL_NAME/decoder_epoch_${CKPT_NUM}.pth
    
    if [ ! -e $MODEL_PATH ]; then
    mkdir -p $MODEL_PATH
    fi
    if [ ! -e $ENCODER ]; then
    wget -P $MODEL_PATH http://sceneparsing.csail.mit.edu/model/pytorch/$ENCODER
    fi
    if [ ! -e $DECODER ]; then
    wget -P $MODEL_PATH http://sceneparsing.csail.mit.edu/model/pytorch/$DECODER
    fi
done