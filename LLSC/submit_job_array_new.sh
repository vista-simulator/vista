#!/bin/bash

#SBATCH -o job_array.sh.log-%j-%a
#SBATCH -c 40
#SBATCH --gres=gpu:volta:2
#SBATCH -a 0-7

# [Check this!!!!] Arguments
BASE_CONFIG=config/local/overtaking-lstm-garage.yaml
JOB_ARRAY_MODULE=search_overtaking_simple
CONVERT_ID=false # NOTE: always check this 
INCLUDE_IDS="0"

echo "My task ID: " $SLURM_ARRAY_TASK_ID
echo "Number of Tasks: " $SLURM_ARRAY_TASK_COUNT
echo "Base config: ${BASE_CONFIG}"
echo "Job array module: ${JOB_ARRAY_MODULE}"

# Make soft link to local filesystem
IFS='/' read -ra ADDR <<< "$TMPDIR"
for i in "${ADDR[@]}"; do
    TMPDIR_ID="$i"
done
ln -s $TMPDIR $HOME/tmp/$TMPDIR_ID
export RAY_TMPDIR=$HOME/tmp/$TMPDIR_ID

# Initialize and Load Modules
source /etc/profile
module unload anaconda
module load anaconda/2020b

# Set variables for code
export PYOPENGL_PLATFORM=egl
# export EGL_DEVICE_ID=$GPU_DEVICE_ORDINAL
export SEG_PRETRAINED_ROOT=$HOME/workspace/semantic-segmentation-pytorch

# [Check this!!!!] Copy data to temporary local filesystem
export DATA_DIR=$TMPDIR/data

export TRACE_DIR=$DATA_DIR/traces
if [ ! -d "$TRACE_DIR" ]; then
    mkdir -p $TRACE_DIR
    cp -r $HOME/data/traces/20210608-115814_lexus_garage $TRACE_DIR
    cp -r $HOME/data/traces/20210608-120054_lexus_garage $TRACE_DIR
    cp -r $HOME/data/traces/20210609-204441_lexus_garage $TRACE_DIR
else
    printf "Trace directory already exists. Wait for 30 sec."
    sleep 30
fi

if [ ! -d "$DATA_DIR/carpack01" ]; then
    cp -r $HOME/data/carpack01 $DATA_DIR
else
    printf "Mesh directory already exists. Wait for 30 sec."
    sleep 30
fi

# Run
cd ${HOME}/workspace/vista/learning
if $CONVERT_ID; then
  ID_LIST=($INCLUDE_IDS)
  TASK_ID=${ID_LIST[${SLURM_ARRAY_TASK_ID}]}
else
  TASK_ID=${SLURM_ARRAY_TASK_ID}
fi
python train_with_job_array.py -f ${BASE_CONFIG} --job-array-module ${JOB_ARRAY_MODULE} --job-array-task-id ${TASK_ID}

# Cleanup
rm $RAY_TMPDIR