# Super-SloMo

This code is built on top of an [addon](https://github.com/prophesee-ai/Super-SloMo) of [Super-SloMo](https://github.com/avinashpaliwal/Super-SloMo) from [Prophsee AI](https://www.prophesee.ai/) to extract optical flow and perform video interpolation.

## Prerequisite
* numpy (tested on 1.21.0)
* torch (tested on 1.8.1+cu102)
* torchvision (tested on 0.9.1+cu102)
* scikit-video (tested on 1.1.11)
* fire (tested on 0.4.0)
* tqdm (tested on 4.60.0)
* opencv-python (tested on 4.5.2)
* h5py (tested on 3.2.1)

## How To Run

First, download model trained on Adobe240fps dataset by `avinashpaliwal` at [here](https://drive.google.com/file/d/1IvobLDbRiBgZr3ryCRrWL8xDbMZ-KnpF/view).

Regular video interpolation.
```
python async_slomo.py <path-to-input-video.mp4> <path-to-output-video.mp4> --checkpoint <path-to-ckpt> --sf 8 --video_fps 30 --viz 1
```

Simple event-based camera simulation based on [this paper](https://arxiv.org/pdf/1912.03095.pdf) with noiseless measurement.
```
python async_slomo.py <path-to-input-video.mp4> <path-to-output-video.mp4> --checkpoint <path-to-ckpt> --sf -1 --video_fps 30
# Check the video with "_dvs" suffix in the directory of the output video path --run-dvs 1
```

Extract optical flow only.
```
python async_slomo.py <path-to-input-video.mp4> <path-to-output-flow.h5> --checkpoint <path-to-ckpt> --extract-flow
```