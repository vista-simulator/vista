import argparse
import cv2
import functools
import numpy as np
import h5py
from skimage.measure import block_reduce
import tensorflow as tf
import os
from tqdm import tqdm
import shutil
from pathlib import Path

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

save_name = "LidarRenderModel.tf"

print("Loading data")
f = h5py.File(os.path.join(args.input, "lidar_3d_vista_big.h5"), "r")



def load_data(mask_name, dense_name):
    MASK = f[mask_name][:]
    DENSE = f[dense_name][:].astype(np.float32)
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

    return mask_train, dense_train, mask_test, dense_test


masko_train, denseo_train, masko_test, denseo_test = load_data(
    "mask_orig", "d_depth_orig")
maskt_train, denset_train, maskt_test, denset_test = load_data(
    "mask_trans", "d_depth_trans")


print("Building model")
optimizer = tf.keras.optimizers.Adam(args.learning_rate)
model = LidarRenderModel(input_shape=masko_train.shape[1:],
                         filters=args.num_filters,
                         num_layers=args.num_layers,
                         dropout=args.dropout)

Path(save_name).mkdir(parents=True, exist_ok=True)
shutil.copy("model.py", save_name)
with open(os.path.join(save_name, "config"), 'w') as f:
    f.write(str(model.get_config()))


def compute_weights(mask, block_size=40):
    # Compute weights
    mask_blocks = block_reduce(mask,
                               block_size=(1, block_size, 1, 1),
                               func=np.mean)
    mask_blocks = mask_blocks.mean((0, 2, 3), keepdims=True)
    mask_blocks = mask_blocks.repeat(block_size, axis=1)
    mask_blocks = (mask_blocks - 0.5) / 1.1 + 0.5
    return mask_blocks


w_mask = compute_weights(masko_train)


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
def train_step(sparse_t, dense_t, dense_o, mask_o):
    with tf.GradientTape() as tape:
        dense_t_ = model.s2d(sparse_t)
        mask_o_ = model.d2mask(dense_o)
        loss, _ = compute_loss(dense_t, mask_o, dense_t_, mask_o_)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return dense_t_, mask_o_, loss


best_vloss = np.float('inf')
alpha = 0.9

sov, mov, dov = sample(masko_train, denseo_train)
dov_, mov_ = model(sov)
vloss_, _ = compute_loss(dov, mov, dov_, mov_)
vloss_ = vloss_.numpy()

pbar = tqdm(range(10000))
for iter in pbar:
    so, mo, do = sample(masko_train, denseo_train)
    st, mt, dt = sample(maskt_train, denset_train)
    dt_, mo_, loss = train_step(st, dt, do, mo)

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
            gray2color(transform(st[0])),
            gray2color(transform(dt[0])),
            gray2color(transform(dt_[0].numpy())),
            gray2color(mo[0]),
            gray2color(prob2mask(mo_[0].numpy()))
        ))
        cv2.imshow("vis", cv2.resize(vis, None, fx=0.75, fy=0.75))
        cv2.waitKey(1)

    if iter % 200 == 0:
        # sv, mv, dv = sample(mask_test, dense_test)
        # dv_, mv_ = model(sv)
        # vloss, _ = compute_loss(dv, mv, dv_, mv_)

        sov, mov, dov = sample(masko_train, denseo_train)
        dov_, mov_ = model(sov)
        vloss, _ = compute_loss(dov, mov, dov_, mov_)
        vloss = vloss.numpy()

        vloss_ = alpha * vloss_ + (1 - alpha) * vloss

        if vloss_ < best_vloss:
            best_vloss = vloss_
            model.save_weights(os.path.join(save_name, "weights.h5"))

        pbar.set_description(f"Loss: {vloss_:.2f} ({best_vloss:.2f})")

model.save_weights(os.path.join(save_name, "weights.h5"))

import pdb
pdb.set_trace()
