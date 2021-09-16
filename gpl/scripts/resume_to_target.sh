#!/bin/bash

TGT_CKPT="iter-0-020-000.pt"
CFG_ROOT="./config/different_envs/event"
RESULT_ROOT="$HOME/results/202109w2/different_envs/event_new_video_labels"
declare -a EXP_ARRAY=("il_rural" "il_urban" "il_daytime_and_night" "il_dry_and_rain")

for EXP in "${EXP_ARRAY[@]}"; do
    CKPT_DIR="$RESULT_ROOT/$EXP/ckpt"
    while ! [ -f "$CKPT_DIR/$TGT_CKPT" ]; do
        python train.py --config $CFG_ROOT/$EXP.yaml --logdir $RESULT_ROOT/$EXP --resume
    done
done

TGT_CKPT="iter-0-025-000.pt"
for EXP in "${EXP_ARRAY[@]}"; do
    CKPT_DIR="$RESULT_ROOT/$EXP/ckpt"
    while ! [ -f "$CKPT_DIR/$TGT_CKPT" ]; do
        python train.py --config $CFG_ROOT/$EXP.yaml --logdir $RESULT_ROOT/$EXP --resume
    done
done

TGT_CKPT="iter-0-030-000.pt"
for EXP in "${EXP_ARRAY[@]}"; do
    CKPT_DIR="$RESULT_ROOT/$EXP/ckpt"
    while ! [ -f "$CKPT_DIR/$TGT_CKPT" ]; do
        python train.py --config $CFG_ROOT/$EXP.yaml --logdir $RESULT_ROOT/$EXP --resume
    done
done

TGT_CKPT="iter-0-035-000.pt"
for EXP in "${EXP_ARRAY[@]}"; do
    CKPT_DIR="$RESULT_ROOT/$EXP/ckpt"
    while ! [ -f "$CKPT_DIR/$TGT_CKPT" ]; do
        python train.py --config $CFG_ROOT/$EXP.yaml --logdir $RESULT_ROOT/$EXP --resume
    done
done

TGT_CKPT="iter-0-040-000.pt"
for EXP in "${EXP_ARRAY[@]}"; do
    CKPT_DIR="$RESULT_ROOT/$EXP/ckpt"
    while ! [ -f "$CKPT_DIR/$TGT_CKPT" ]; do
        python train.py --config $CFG_ROOT/$EXP.yaml --logdir $RESULT_ROOT/$EXP --resume
    done
done
