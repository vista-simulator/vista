#!/bin/bash

#SBATCH -o submit_job_array.sh.log-%j-%a
#SBATCH -c 20
#SBATCH --gres=gpu:volta:1
#SBATCH -a 1-12

USE_EGL=true
BASE_CONFIG=config/takeover-ppo/with-velocity.yaml
JOB_ARRAY_MODULE=search_sgd_params

echo "USE_EGL: ${USE_EGL}"
echo "BASE_CONFIG: ${BASE_CONFIG}"
echo "JOB_ARRAY_MODULE: ${JOB_ARRAY_MODULE}"
echo "TASK ID: ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}"

if $USE_EGL
then
  export PYOPENGL_PLATFORM=egl
  export EGL_DEVICE_ID=$(nvidia-smi -q | grep "Minor Number" | cut -f 1 -d$'\n' | awk '{print substr($NF,0,1)}' | tail -1)
else
  startx &
  sleep 5
  DISPLAYNUM=$(ps -efww | grep xinit | grep -v grep | sed -re "s/^.* :([0-9]+) .*$/\1/g" | tail -1)
  export DISPLAY=:${DISPLAYNUM}
  glxinfo
fi

eval "$(conda shell.bash hook)"
conda activate vista

cd ${HOME}/workspace/vista-integrate-new-api/learning
python -m job_array.${JOB_ARRAY_MODULE} ${SLURM_ARRAY_TASK_ID} # print task-specific config
python train_with_job_array.py -f ${BASE_CONFIG} --job-array-module ${JOB_ARRAY_MODULE} --job-array-task-id ${SLURM_ARRAY_TASK_ID}