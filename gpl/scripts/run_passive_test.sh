#!/bin/bash

CKPT_ROOT_DIR="$HOME/results/202109w2/different_envs/event_new_video_labels"
TEST_CFG_ROOT="./config/different_envs/event_test"
declare -a CKPT_ARRAY=("iter-0-120-000.pt")
declare -a TEST_CFG_ARRAY=("daytime" "night" "rain" "dry" "rural" "urban")

for MODEL_DIR in $CKPT_ROOT_DIR/*; do
    if [ -d "$MODEL_DIR" ]; then
        for CKPT in "${CKPT_ARRAY[@]}"; do
            CKPT_PATH=$MODEL_DIR/ckpt/$CKPT
            if [ -f "$CKPT_PATH" ]; then
                for TEST_CFG in "${TEST_CFG_ARRAY[@]}"; do
                    if [[ "$CKPT_PATH" == *"$TEST_CFG"* ]]; then
                        python test.py --ckpt $CKPT_PATH --mode passive --test-config $TEST_CFG_ROOT/passive_$TEST_CFG.yaml
                    fi
                done
            fi
        done
    fi
done