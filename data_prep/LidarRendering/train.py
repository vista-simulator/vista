import cv2
import numpy as np
from keras_unet.models import custom_unet
from skimage.measure import block_reduce
import tensorflow as tf

data = np.load("dense_lidar.npz")
x = np.expand_dims(data['dist'], -1)[:, :-2, :-1, :]
x = x.astype(np.float32)
y = np.expand_dims(data['mask'], -1)[:, :-2, :-1, :]
y = y.astype(np.float32)
del data

learning_rate = 1e-4
batch_size = 2

optimizer = tf.keras.optimizers.Adam(learning_rate)
model = custom_unet(input_shape=x.shape[1:],
                    num_classes=1,
                    filters=16,
                    num_layers=4,
                    dropout=0.05,
                    output_activation='sigmoid')


@tf.function
def train_step(xx, yy, w):
    with tf.GradientTape() as tape:
        y_hat = model(xx)
        loss = (1 - w) * yy * tf.math.log(y_hat + 1e-7) \
             + w * (1 - yy) * tf.math.log(1 - y_hat + 1e-7)
        loss = 2 * loss
        loss = tf.reduce_mean(-1 * loss)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


def sample():
    i = np.random.choice(x.shape[0], batch_size)
    k = np.random.choice(x.shape[2], 1)[0]
    xx = np.roll(x[i], k, axis=1)
    yy = np.roll(y[i], k, axis=1)
    return xx, yy


for iter in range(10000):
    xx, yy = sample()
    yy_blocks = block_reduce(yy, block_size=(1, 50, 1, 1), func=np.mean)
    yy_blocks = yy_blocks.mean((0, 2, 3), keepdims=True)
    yy_blocks = yy_blocks.repeat(50, axis=1)
    yy_blocks = (yy_blocks - 0.5) / 1.1 + 0.5

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
