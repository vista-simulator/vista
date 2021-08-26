import argparse
import cv2
import numpy as np
import h5py
from keras_unet.models import custom_unet
from skimage.measure import block_reduce
import tensorflow as tf
import os
from tqdm import tqdm

# Parse Arguments
parser = argparse.ArgumentParser(
    description='Train masking network for lidar rendering.')
parser.add_argument('-i',
                    '--input',
                    type=str,
                    required=True,
                    help='Path to the trace to train with')
parser.add_argument('-b', '--batch_size', type=int, default=2)
parser.add_argument('-r', '--learning_rate', type=float, default=1e-3)
parser.add_argument('-p', '--dropout', type=float, default=0.05)
parser.add_argument('-l', '--num_layers', type=int, default=3)
args = parser.parse_args()

print("Loading data")
f = h5py.File(os.path.join(args.input, "lidar_3d_vista_new.h5"), "r")

# only train with a small subset (debugging for speed)
# dataset_size = f["d_depth"].shape[0]
# data_idx = np.random.choice(f["d_depth"].shape[0], dataset_size, replace=False)
# data_idx = np.sort(data_idx)

x = f['mask_trans'][:50]
yd = f['d_depth_trans'][:50].astype(np.float32)
yi = f['d_int_trans'][:50].astype(np.uint8)
train_idx = np.random.choice(x.shape[0], int(x.shape[0] * 0.8), replace=False)
test_idx = np.setdiff1d(np.arange(x.shape[0]), train_idx, assume_unique=True)
x_train, yd_train, yi_train, x_test, yd_test, yi_test = (x[train_idx],
                                                         yd[train_idx],
                                                         yi[train_idx],
                                                         x[test_idx],
                                                         yd[test_idx],
                                                         yi[train_idx])
del x, yd, yi

print("Building model")
optimizer = tf.keras.optimizers.Adam(args.learning_rate)

_, h, w, _ = x_train.shape
UNET = custom_unet(
    input_shape=(h, w, 3),
    num_classes=2,
    filters=16,
    num_layers=args.num_layers,
    dropout=args.dropout,
    upsample_mode="deconv",
    # activation="swish",
    output_activation=None)

Lambda = tf.keras.layers.Lambda
model = tf.keras.Sequential([
    Lambda(lambda x: x / tf.constant([50., 255, 1])),
    UNET,
    Lambda(lambda y: tf.stack((tf.math.exp(y[..., 0] - tf.math.log(150.) / 2),
                               tf.math.exp(y[..., 1] - tf.math.log(255.) / 2)),
                              axis=-1)),
    Lambda(lambda y: y * tf.constant([50., 255])),
])


@tf.function
def compute_loss(yy, y_hat, k=50.):
    i = yy > 0.
    loss = tf.abs((yy[i] / k) - (y_hat[i] / k))
    loss = tf.reduce_mean(loss)
    return loss


@tf.function
def train_step(xx, yy):
    with tf.GradientTape() as tape:
        y_hat = model(xx)
        loss = compute_loss(yy, y_hat)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return y_hat, loss


def sample(X, Yd, Yi):
    i = np.random.choice(X.shape[0], args.batch_size)
    k = np.random.choice(X.shape[2], 1)[0]

    mask, yd, yi = (X[i], Yd[i], Yi[i])
    yy = np.concatenate((yd, yi), axis=3)
    xx = np.concatenate((yy, mask), axis=3)
    xx[~mask[..., 0], :] = 0.0

    xx = np.roll(xx, k, axis=2).astype(np.float32)
    yy = np.roll(yy, k, axis=2).astype(np.float32)
    return xx, yy


best_vloss = np.float('inf')
alpha = 0.9

xv, yv = sample(x_test, yd_test, yi_test)
yv_hat = model(xv)
vloss = compute_loss(yv, yv_hat).numpy()

pbar = tqdm(range(10000))
for iter in pbar:
    xx, yy = sample(x_train, yd_train, yi_train)
    yy_hat, loss = train_step(xx, yy)
    # print(yy_hat.numpy().min(), yy_hat.numpy().mean(), yy_hat.numpy().max())

    if iter % 20 == 0:

        def gray2color(gray, scaling=70.):
            gray = np.clip(gray / scaling * 255, 0, 255).astype(np.uint8)
            color = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
            return color

        out = (
            gray2color(xx[0, :, :, 0]),
            gray2color(yy[0, :, :, 0]),
            gray2color(yy_hat[0, :, :, 0].numpy()),
            gray2color(xx[0, :, :, 1]),
            gray2color(yy[0, :, :, 1]),
            gray2color(yy_hat[0, :, :, 1].numpy()),
        )
        cv2.imshow("out",
                   cv2.resize(np.concatenate(out, 0), None, fx=0.5, fy=0.5))
        # cv2.imshow('pred', gray2color(yy_hat[0, :, :, 0].numpy()))
        # cv2.imshow('true', gray2color(yy[0, :, :, 0]))
        # cv2.imshow('input', gray2color(xx[0, :, :, 0]))
        cv2.waitKey(1)

    if iter % 50 == 0:
        xv, yv = sample(x_test, yd_test, yi_test)
        yv_hat = model(xv)
        vloss_ = compute_loss(yv, yv_hat).numpy()
        vloss = alpha * vloss + (1 - alpha) * vloss_

        if vloss < best_vloss:
            best_vloss = vloss
            tf.keras.models.save_model(model, "LidarFiller3.h5")

        pbar.set_description(f"Loss: {vloss:.2f} ({best_vloss:.2f})")

tf.keras.models.save_model(model, "LidarFiller3.h5")

import pdb
pdb.set_trace()
