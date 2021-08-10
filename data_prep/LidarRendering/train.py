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
parser.add_argument('-r', '--learning_rate', type=float, default=1e-4)
parser.add_argument('-p', '--dropout', type=float, default=0.05)
parser.add_argument('-l', '--num_layers', type=int, default=3)
args = parser.parse_args()

print("Loading data")
f = h5py.File(os.path.join(args.input, "lidar_3d_vista.h5"), "r")

# only train with a small subset (debugging for speed)
# dataset_size = f["d_depth"].shape[0]
# data_idx = np.random.choice(f["d_depth"].shape[0], dataset_size, replace=False)
# data_idx = np.sort(data_idx)

x = f['d_depth'][:].astype(np.float32) / 50.
y = f['mask'][:].astype(np.float32)
train_idx = np.random.choice(x.shape[0], int(x.shape[0] * 0.8), replace=False)
test_idx = np.setdiff1d(np.arange(x.shape[0]), train_idx, assume_unique=True)
x_train, y_train, x_test, y_test = (x[train_idx], y[train_idx], x[test_idx],
                                    y[test_idx])
del x, y

print("Building model")
optimizer = tf.keras.optimizers.Adam(args.learning_rate)
model = custom_unet(input_shape=x_train.shape[1:],
                    num_classes=1,
                    filters=16,
                    num_layers=args.num_layers,
                    dropout=args.dropout,
                    output_activation='sigmoid')


@tf.function
def compute_loss(yy, y_hat, w):
    loss = (1 - w) * yy * tf.math.log(y_hat + 1e-7) \
         + w * (1 - yy) * tf.math.log(1 - y_hat + 1e-7)
    loss = -2 * loss
    loss = tf.reduce_mean(loss)
    return loss


@tf.function
def train_step(xx, yy, w):
    with tf.GradientTape() as tape:
        y_hat = model.call(xx)
        loss = compute_loss(yy, y_hat, w)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


def compute_weights(yy, block_size=60):
    # Compute weights
    yy_blocks = block_reduce(yy,
                             block_size=(1, block_size, 1, 1),
                             func=np.mean)
    yy_blocks = yy_blocks.mean((0, 2, 3), keepdims=True)
    yy_blocks = yy_blocks.repeat(block_size, axis=1)
    yy_blocks = (yy_blocks - 0.5) / 1.1 + 0.5
    return yy_blocks


def sample():
    i = np.random.choice(x_train.shape[0], args.batch_size)
    k = np.random.choice(x_train.shape[2], 1)[0]
    xx = np.roll(x_train[i], k, axis=1).astype(np.float32)
    yy = np.roll(y_train[i], k, axis=1).astype(np.float32)
    yy_blocks = compute_weights(yy)
    return xx, yy, yy_blocks
    # return (tf.convert_to_tensor(xx), \
    #         tf.convert_to_tensor(yy),
    #         tf.convert_to_tensor(yy_blocks))


best_vloss = np.float('inf')
vloss = compute_loss(y_test[[0]].astype(np.float32),
                     model.predict(x_test[[0]].astype(np.float32)),
                     compute_weights(y_test[[0]]).astype(np.float32)).numpy()
alpha = 0.95

pbar = tqdm(range(10000))
for iter in pbar:
    xx, yy, yy_blocks = sample()
    loss = train_step(xx, yy, yy_blocks)

    if iter % 20 == 0:
        yv_hat = model.predict(x_train[[0]])[0]
        thresh = np.percentile(yv_hat, 80)

        cv2.imshow('pred', (yv_hat > thresh).astype(np.float32))
        cv2.imshow('true', y_train[0].astype(np.float32))
        cv2.waitKey(1)

    if iter % 50 == 0:
        vind = np.random.choice(x_test.shape[0], args.batch_size)
        yv_hat = model.predict(x_test[vind].astype(np.float32))
        v_weights = compute_weights(y_test[vind]).astype(np.float32)
        vloss_ = compute_loss(y_test[vind].astype(np.float32), yv_hat,
                              v_weights).numpy()
        vloss = alpha * vloss + (1 - alpha) * vloss_

        if vloss < best_vloss:
            best_vloss = vloss
            tf.keras.models.save_model(model, "LidarMaskModel.h5")

        pbar.set_description(f"Loss: {vloss:.2f} ({best_vloss:.2f})")

tf.keras.models.save_model(model, "LidarMaskModel.h5")

import pdb
pdb.set_trace()
