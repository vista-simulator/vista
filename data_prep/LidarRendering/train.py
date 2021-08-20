import argparse
import cv2
import functools
import numpy as np
import h5py
from skimage.measure import block_reduce
import tensorflow as tf
import os
from tqdm import tqdm

from model import LidarRenderModel

# Parse Arguments
parser = argparse.ArgumentParser(
    description='Train masking network for lidar rendering.')
parser.add_argument('-i',
                    '--input',
                    type=str,
                    required=True,
                    help='Path to the trace to train with')
parser.add_argument('-b', '--batch_size', type=int, default=2)
parser.add_argument('-r', '--learning_rate', type=float, default=1e-4)
parser.add_argument('-p', '--dropout', type=float, default=0.05)
parser.add_argument('-l', '--num_layers', type=int, default=3)
parser.add_argument('-f', '--num_filters', type=int, default=8)
args = parser.parse_args()

save_name = "LidarRenderModel__.tf"

print("Loading data")
f = h5py.File(os.path.join(args.input, "lidar_3d_vista.h5"), "r")

# only train with a small subset (debugging for speed)
# dataset_size = f["d_depth"].shape[0]
# data_idx = np.random.choice(f["d_depth"].shape[0], dataset_size, replace=False)
# data_idx = np.sort(data_idx)

MASK = f['mask'][:10]
DENSE = f['d_depth'][:10].astype(np.float32)
train_idx = np.random.choice(MASK.shape[0],
                             int(MASK.shape[0] * 0.8),
                             replace=False)
test_idx = np.setdiff1d(np.arange(MASK.shape[0]),
                        train_idx,
                        assume_unique=True)
mask_train = MASK[train_idx]
dense_train = DENSE[train_idx]
mask_test = MASK[test_idx]
dense_test = DENSE[test_idx]
del MASK, DENSE

print("Building model")
optimizer = tf.keras.optimizers.Adam(args.learning_rate)
model = LidarRenderModel(input_shape=mask_train.shape[1:],
                         filters=args.num_filters,
                         num_layers=args.num_layers,
                         dropout=args.dropout)


def compute_weights(mask, block_size=40):
    # Compute weights
    mask_blocks = block_reduce(mask,
                               block_size=(1, block_size, 1, 1),
                               func=np.mean)
    mask_blocks = mask_blocks.mean((0, 2, 3), keepdims=True)
    mask_blocks = mask_blocks.repeat(block_size, axis=1)
    mask_blocks = (mask_blocks - 0.5) / 1.1 + 0.5
    return mask_blocks


w_mask = compute_weights(mask_train)


def sample(mask, dense):
    i = np.random.choice(mask.shape[0], args.batch_size)
    k = np.random.choice(mask.shape[2], 1)[0]

    _mask, _dense = (mask[i], dense[i])
    _sparse = np.copy(_dense)
    _sparse[~_mask] = 0.0

    _sparse = np.roll(_sparse, k, axis=2).astype(np.float32)
    _dense = np.roll(_dense, k, axis=2).astype(np.float32)
    _mask = np.roll(_mask, k, axis=2).astype(np.float32)
    return _sparse, _mask, _dense


@tf.function
def compute_loss(dense, mask, dense_pred, mask_pred, coeff=1 / 50., eps=1e-7):
    valid = dense > 0.
    dense_loss = tf.reduce_mean(tf.abs(dense[valid] - dense_pred[valid]))

    mask_loss = (1 - w_mask) * mask * tf.math.log(mask_pred + eps) \
                + w_mask * (1 - mask) * tf.math.log(1 - mask_pred + eps)
    mask_loss = tf.reduce_mean(-2 * mask_loss)

    loss = mask_loss + coeff * dense_loss
    return loss, (dense_loss, mask_loss)


@tf.function
def train_step(sparse, mask, dense):
    with tf.GradientTape() as tape:
        dense_ = model.s2d(sparse)
        mask_ = model.d2mask(dense)
        loss, _ = compute_loss(dense, mask, dense_, mask_)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return dense_, mask_, loss


best_vloss = np.float('inf')
alpha = 0.9

sv, mv, dv = sample(mask_test, dense_test)
dv_, mv_ = model(sv)
vloss_, _ = compute_loss(dv, mv, dv_, mv_)
vloss_ = vloss_.numpy()

pbar = tqdm(range(10000))
for iter in pbar:
    ss, mm, dd = sample(mask_train, dense_train)
    dd_, mm_, loss = train_step(ss, mm, dd)

    if iter % 20 == 0:

        def gray2color(gray):
            gray = np.clip(gray * 255, 0, 255).astype(np.uint8)
            color = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
            return color

        def transform(x, k=5, eps=1e-7):
            return np.log(x + eps) / k

        def prob2mask(p, percentile=80):
            thresh = np.percentile(p, percentile)
            return (p > thresh).astype(np.float32)

        # import pdb; pdb.set_trace()
        vis = np.vstack(( \
            gray2color(transform(ss[0])),
            gray2color(transform(dd[0])),
            gray2color(transform(dd_[0].numpy())),
            gray2color(mm[0]),
            gray2color(prob2mask(mm_[0].numpy()))
        ))
        cv2.imshow("vis", cv2.resize(vis, None, fx=0.5, fy=0.5))
        cv2.waitKey(1)

    if iter % 200 == 0:
        sv, mv, dv = sample(mask_test, dense_test)
        dv_, mv_ = model(sv)
        vloss, _ = compute_loss(dv, mv, dv_, mv_)
        vloss = alpha * vloss + (1 - alpha) * vloss_

        if vloss < best_vloss:
            best_vloss = vloss
            # tf.keras.models.save_model(model, save_name, save_format="tf")
            tf.keras.models.save_model(model, "test", save_format="tf")
            # tf.saved_model.save(model, "test")
            model.save_weights(os.path.join("test", "weights.h5"))



        pbar.set_description(f"Loss: {vloss:.2f} ({best_vloss:.2f})")

tf.keras.models.save_model(model, save_name, save_format="tf")

import pdb
pdb.set_trace()
