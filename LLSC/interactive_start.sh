"""run source interactive_start.sh"""
#!/bin/bash

USE_EGL=true

export SEG_PRETRAINED_ROOT=$HOME/workspace/semantic-segmentation-pytorch

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