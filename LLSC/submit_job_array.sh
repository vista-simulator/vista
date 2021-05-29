#!/bin/bash

#SBATCH -o submit_job_array.sh.log-%j-%a
#SBATCH -c 20
#SBATCH --gres=gpu:volta:1
#SBATCH -a 1-1

USE_GPU=true
USE_EGL=true
BASE_CONFIG=config/state-obs-ppo/bev-cutting-off.yaml
JOB_ARRAY_MODULE=search_example
CONVERT_ID=false # NOTE: always check this 
INCLUDE_IDS="4"

export SEG_PRETRAINED_ROOT=$HOME/workspace/semantic-segmentation-pytorch

echo "USE_GPU: ${USE_GPU}"
echo "USE_EGL: ${USE_EGL}"
echo "BASE_CONFIG: ${BASE_CONFIG}"
echo "JOB_ARRAY_MODULE: ${JOB_ARRAY_MODULE}"
echo "TASK ID: ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}"

if $USE_GPU
then
  if $USE_EGL
  then
    export PYOPENGL_PLATFORM=egl
    export EGL_DEVICE_ID=$GPU_DEVICE_ORDINAL
  else
    startx &
    sleep 5
    DISPLAYNUM=$(ps -efww | grep xinit | grep -v grep | sed -re "s/^.* :([0-9]+) .*$/\1/g" | tail -1)
    export DISPLAY=:${DISPLAYNUM}
    glxinfo
  fi

  eval "$(conda shell.bash hook)"
  conda activate vista
else # cpu-only rendering
  export PYOPENGL_PLATFORM=osmesa

  eval "$(conda shell.bash hook)"
  conda activate vista_cpu
fi

cd ${HOME}/workspace/vista-integrate-new-api/learning
if $CONVERT_ID; then
  ID_LIST=($INCLUDE_IDS)
  NEW_TASK_ID=${ID_LIST[${SLURM_ARRAY_TASK_ID}]}
  python train_with_job_array.py -f ${BASE_CONFIG} --job-array-module ${JOB_ARRAY_MODULE} --job-array-task-id ${NEW_TASK_ID}
else
  python train_with_job_array.py -f ${BASE_CONFIG} --job-array-module ${JOB_ARRAY_MODULE} --job-array-task-id ${SLURM_ARRAY_TASK_ID}
fi