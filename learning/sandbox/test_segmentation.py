import os
import sys
import argparse
from ffio import FFReader
import numpy as np
from scipy.io import loadmat
import csv
from PIL import Image
import cv2
import torch
import torch.nn as nn
from torchvision import transforms

from vista.util.Camera import Camera

sys.path.insert(0, os.environ.get('SEG_PRETRAINED_ROOT'))
try:
    from mit_semseg.models import ModelBuilder, SegmentationModule
    from mit_semseg.config import cfg
    from mit_semseg.utils import colorEncode
    from mit_semseg.lib.utils import as_numpy
except ImportError:
    raise ImportError('Fail to import segmentation repo. Do you forget to set SEG_PRETRAINED_ROOT ?')

IMAGE_SIZE=(200, 320)

### parse arguments
parser = argparse.ArgumentParser('PyTorch Semantic Segmentation Testing')
parser.add_argument('--seg-model-name', type=str, required=True)
parser.add_argument('--video', type=str, required=True)
parser.add_argument('--frame-num', type=int, default=0)
parser.add_argument('--cuda', action='store_true', default=False)
parser.add_argument('--roi', action='store_true', default=False)
parser.add_argument('--resize', type=float, default=None)
args = parser.parse_args()

### set up segmentation model
config_arg = os.path.join(
    os.environ.get('SEG_PRETRAINED_ROOT'),
    'config/{}.yaml'.format(args.seg_model_name))
cfg.merge_from_file(config_arg)
cfg.MODEL.arch_encoder = cfg.MODEL.arch_encoder.lower()
cfg.MODEL.weights_encoder = os.path.join(
    os.environ.get('SEG_PRETRAINED_ROOT'),
    cfg.DIR, 'encoder_' + cfg.TEST.checkpoint)
cfg.MODEL.weights_decoder = os.path.join(
    os.environ.get('SEG_PRETRAINED_ROOT'),
    cfg.DIR, 'decoder_' + cfg.TEST.checkpoint)
assert os.path.exists(cfg.MODEL.weights_encoder) and \
    os.path.exists(cfg.MODEL.weights_decoder), 'checkpoint does not exitst!'

net_encoder = ModelBuilder.build_encoder(
    arch=cfg.MODEL.arch_encoder,
    fc_dim=cfg.MODEL.fc_dim,
    weights=cfg.MODEL.weights_encoder)
net_decoder = ModelBuilder.build_decoder(
    arch=cfg.MODEL.arch_decoder,
    fc_dim=cfg.MODEL.fc_dim,
    num_class=cfg.DATASET.num_class,
    weights=cfg.MODEL.weights_decoder,
    use_softmax=True)
if False:
    for module in net_encoder.modules():
        if isinstance(module, nn.Conv2d):
            if module.dilation[0] != 1:
                n = module.dilation[0]
                new_dilation = module.dilation[0] // n
                new_padding = int(module.padding[0] - module.dilation[0] * (module.kernel_size[0] - 1) / 2 * (1 - 1 / n))
                module.dilation = (new_dilation, new_dilation)
                module.padding = (new_padding, new_padding)
crit = nn.NLLLoss(ignore_index=-1)
segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
if args.cuda:
    segmentation_module.cuda()
segmentation_module.eval()

### functions that converts segmentation map to color-coded visualization
colors = loadmat(os.path.join(os.environ.get('SEG_PRETRAINED_ROOT'), 'data/color150.mat'))['colors']
names = {}
with open(os.path.join(os.environ.get('SEG_PRETRAINED_ROOT'), 'data/object150_info.csv')) as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        names[int(row[0])] = row[5].split(";")[0]

def visualize_result(img, pred):
    # print predictions in descending order
    pred = np.int32(pred)
    pixs = pred.size
    uniques, counts = np.unique(pred, return_counts=True)
    for idx in np.argsort(counts)[::-1]:
        name = names[uniques[idx] + 1]
        ratio = counts[idx] / pixs * 100
        if ratio > 0.1:
            print("  {}: {:.2f}%".format(name, ratio))

    # colorize prediction
    pred_color = colorEncode(pred, colors).astype(np.uint8)

    # aggregate images and save
    im_vis = np.concatenate((img, pred_color), axis=1)

    return im_vis

### get image
args.video = os.path.expanduser(args.video)
stream = FFReader(args.video, custom_size=IMAGE_SIZE, verbose=False)
seek_sec = stream.frame_to_secs(args.frame_num)
stream.seek(seek_sec)
stream.read()
img = stream.image.copy()

cam_name = os.path.basename(args.video).split('.')[0]
camera = Camera(cam_name)
camera.resize(*IMAGE_SIZE)
if args.roi:
    (i1, j1, i2, j2) = camera.get_roi()
    img = img[i1:i2, j1:j2]
if args.resize:
    img = cv2.resize(img, None, fx=args.resize, fy=args.resize)

### NOTE preprocess data
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])
def img_transform(_img):
    _img = np.float32(np.array(_img.copy())) / 255. # 0-255 to 0-1
    _img = _img.transpose((2, 0, 1))
    _img = normalize(torch.from_numpy(_img))
    return _img
img = img[:,:,::-1] # NOTE RGB format
img_pp = torch.unsqueeze(img_transform(img), 0)
if args.cuda:
    img_pp = img_pp.cuda()

### run inference
segSize = (img_pp.shape[2], img_pp.shape[3])
feed_dict = dict()
feed_dict['img_data'] = img_pp
with torch.no_grad():
    scores = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1])
    if args.cuda:
        scores = scores.cuda()
    pred_tmp = segmentation_module(feed_dict, segSize=segSize)
    scores = scores + pred_tmp / len(cfg.DATASET.imgSizes)
    _, pred = torch.max(scores, dim=1)
    pred = as_numpy(pred.squeeze(0).cpu())
out = visualize_result(img, pred)
Image.fromarray(out).save('test.jpg')
import pdb; pdb.set_trace()