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
f = h5py.File(os.path.join(args.input, "lidar_3d_vista.h5"), "r")

# only train with a small subset (debugging for speed)
# dataset_size = f["d_depth"].shape[0]
# data_idx = np.random.choice(f["d_depth"].shape[0], dataset_size, replace=False)
# data_idx = np.sort(data_idx)

x = f['mask'][:]
y = f['d_depth'][:].astype(np.float32)
train_idx = np.random.choice(x.shape[0], int(x.shape[0] * 0.8), replace=False)
test_idx = np.setdiff1d(np.arange(x.shape[0]), train_idx, assume_unique=True)
x_train, y_train, x_test, y_test = (x[train_idx], y[train_idx], x[test_idx],
                                    y[test_idx])
del x, y

print("Building model")
optimizer = tf.keras.optimizers.Adam(args.learning_rate)


class LidarRenderModel(tf.keras.Model):
    def __init__(self, input_shape):
        super(LidarRenderModel, self).__init__()
        unet = functools.partial(
            custom_unet,
            input_shape=input_shape,
            filters=8,
            num_layers=args.num_layers,
            dropout=args.dropout,
            upsample_mode="deconv",
        )
        self.unet_s2d = unet(num_classes=1, output_activation=None)
        self.unet_mask = unet(num_classes=1, output_activation=None)

        self.block_1 = ResNetBlock()
        self.block_2 = ResNetBlock()
        self.global_pool = layers.GlobalAveragePooling2D()
        self.classifier = Dense(num_classes)

    def call(self, inputs):
        x = self.block_1(inputs)
        x = self.block_2(x)
        x = self.global_pool(x)
        return self.classifier(x)


unet = custom_unet(input_shape=x_train.shape[1:],
                   num_classes=1,
                   filters=8,
                   num_layers=args.num_layers,
                   dropout=args.dropout,
                   upsample_mode="deconv",
                   output_activation=tf.math.exp)

model = tf.keras.Sequential([
    tf.keras.layers.Lambda(lambda x: x / 50.),
    unet,
    tf.keras.layers.Lambda(lambda x: x * 50.),
])


@tf.function
def compute_loss(yy, y_hat):
    i = yy > 0.
    loss = (yy[i] - y_hat[i])**2
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


def sample(X, Y):
    i = np.random.choice(X.shape[0], args.batch_size)
    k = np.random.choice(X.shape[2], 1)[0]

    mask, yy = (X[i], Y[i])
    xx = np.copy(yy)
    xx[~mask] = 0.0

    xx = np.roll(xx, k, axis=2).astype(np.float32)
    yy = np.roll(yy, k, axis=2).astype(np.float32)
    return xx, yy


best_vloss = np.float('inf')
alpha = 0.9

xv, yv = sample(x_test, y_test)
yv_hat = model(xv)
vloss = compute_loss(yv, yv_hat).numpy()

pbar = tqdm(range(10000))
for iter in pbar:
    xx, yy = sample(x_train, y_train)
    yy_hat, loss = train_step(xx, yy)

    if iter % 20 == 0:
        scaling = 50.

        def gray2color(gray):
            gray = np.clip(gray / scaling * 255, 0, 255).astype(np.uint8)
            color = cv2.applyColorMap(gray, cv2.COLORMAP_HSV)
            return color

        cv2.imshow('pred', gray2color(yy_hat[0].numpy()))
        cv2.imshow('true', gray2color(yy[0]))
        cv2.imshow('input', gray2color(xx[0]))
        cv2.waitKey(1)

    if iter % 50 == 0:
        xv, yv = sample(x_test, y_test)
        yv_hat = model(xv)
        vloss_ = compute_loss(yv, yv_hat).numpy()
        vloss = alpha * vloss + (1 - alpha) * vloss_

        if vloss < best_vloss:
            best_vloss = vloss
            tf.keras.models.save_model(model, "LidarFiller2.h5")

        pbar.set_description(f"Loss: {vloss:.2f} ({best_vloss:.2f})")

tf.keras.models.save_model(model, "LidarFiller2.h5")

import pdb
pdb.set_trace()
