#!/bin/bash

# [Check this!!!!] Arguments
BASE_CONFIG=config/local/overtaking-lstm-garage.yaml
JOB_ARRAY_MODULE=search_overtaking_simple
CONVERT_ID=false # NOTE: always check this 
INCLUDE_IDS="4"

echo "My task ID: " $LLSUB_RANK
echo "Number of Tasks: " $LLSUB_SIZE
echo "Base config: ${BASE_CONFIG}"
echo "Job array module: ${JOB_ARRAY_MODULE}"

# Initialize and Load Modules
source /etc/profile
module unload anaconda
module load anaconda/2020b

# Set variables for code
export PYOPENGL_PLATFORM=egl
export EGL_DEVICE_ID=$GPU_DEVICE_ORDINAL
export SEG_PRETRAINED_ROOT=$HOME/workspace/semantic-segmentation-pytorch

# [Check this!!!!] Copy data to temporary local filesystem
export DATA_DIR=$TMPDIR/data

export TRACE_DIR=$DATA_DIR/traces
if [ ! -d "$TRACE_DIR" ] && [ ${LLSUB_RANK} = "1" ]; then
    mkdir -p $TRACE_DIR
    cp -r $HOME/data/traces/20210608-115814_lexus_garage $TRACE_DIR
    cp -r $HOME/data/traces/20210608-120054_lexus_garage $TRACE_DIR
    cp -r $HOME/data/traces/20210609-204441_lexus_garage $TRACE_DIR
else
    printf "Trace directory already exists. Wait for 30 sec."
    sleep 30
fi

if [ ! -d "$DATA_DIR/carpack01" ] && [ ${LLSUB_RANK} = "1" ]; then
    cp -r $HOME/data/carpack01 $DATA_DIR
else
    printf "Mesh directory already exists. Wait for 30 sec."
    sleep 30
fi

# Run
cd ${HOME}/workspace/vista/learning
if $CONVERT_ID; then
  ID_LIST=($INCLUDE_IDS)
  TASK_ID=${ID_LIST[${LLSUB_RANK}]}
else
  TASK_ID=${LLSUB_RANK}
fi
python train_with_job_array.py -f ${BASE_CONFIG} --job-array-module ${JOB_ARRAY_MODULE} --job-array-task-id ${TASK_ID}