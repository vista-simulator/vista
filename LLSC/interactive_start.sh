"""run source interactive_start.sh"""
#!/bin/bash

USE_EGL=false

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