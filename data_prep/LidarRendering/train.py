import argparse
import cv2
import numpy as np
import h5py
from keras_unet.models import custom_unet
from skimage.measure import block_reduce
import tensorflow as tf
import os

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
x = f['dense'][:5, :, :, [0]].astype(np.float32) / 50.
y = f['mask'][:5].astype(np.float32)

print("Building model")
optimizer = tf.keras.optimizers.Adam(args.learning_rate)
model = custom_unet(input_shape=x.shape[1:],
                    num_classes=1,
                    filters=16,
                    num_layers=args.num_layers,
                    dropout=args.dropout,
                    output_activation='sigmoid')


@tf.function
def train_step(xx, yy, w):
    with tf.GradientTape() as tape:
        y_hat = model(xx)
        loss = (1 - w) * yy * tf.math.log(y_hat + 1e-7) \
             + w * (1 - yy) * tf.math.log(1 - y_hat + 1e-7)
        loss = -2 * loss
        loss = tf.reduce_mean(loss)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


def sample(block_size=60):
    i = np.random.choice(x.shape[0], args.batch_size)
    k = np.random.choice(x.shape[2], 1)[0]
    xx = np.roll(x[i], k, axis=1)
    yy = np.roll(y[i], k, axis=1)

    # Compute weights
    yy_blocks = block_reduce(yy,
                             block_size=(1, block_size, 1, 1),
                             func=np.mean)
    yy_blocks = yy_blocks.mean((0, 2, 3), keepdims=True)
    yy_blocks = yy_blocks.repeat(block_size, axis=1)
    yy_blocks = (yy_blocks - 0.5) / 1.1 + 0.5
    return xx, yy, yy_blocks


for iter in range(10000):
    xx, yy, yy_blocks = sample()
    loss = train_step(xx, yy, yy_blocks)
    print(loss.numpy())

    if iter % 20 == 0:
        yv_hat = model.predict(x[[0]])[0]
        print("mean", yv_hat.mean())
        thresh = np.percentile(yv_hat, 80)

        cv2.imshow('pred', (yv_hat > thresh).astype(np.float32))
        cv2.imshow('true', y[0])
        cv2.waitKey(1)
    if iter % 1000 == 0:
        print("saving")
        tf.keras.models.save_model(model, "LidarMaskModel.h5")

tf.keras.models.save_model(model, "LidarMaskModel.h5")

import pdb
pdb.set_trace()
