import functools

import tensorflow as tf
from keras_unet.models import custom_unet


class LidarRenderModel(tf.keras.Model):
    def __init__(self, scale=50., **kwargs):
        super(LidarRenderModel, self).__init__()
        my_unet = functools.partial(custom_unet,
                                    upsample_mode="deconv",
                                    **kwargs)

        self.scale = scale
        self.kwargs = kwargs
        self.unet_s2d = my_unet(num_classes=1, output_activation=tf.math.exp)
        self.unet_mask = my_unet(num_classes=1, output_activation='sigmoid')

    @tf.function
    def call(self, sparse):
        dense_scaled = self.unet_s2d(sparse / self.scale)
        dense_pred = dense_scaled * self.scale

        mask_pred = self.unet_mask(dense_scaled)
        return dense_pred, mask_pred

    @tf.function
    def s2d(self, x):
        dense_scaled = self.unet_s2d(x / self.scale)
        dense = dense_scaled * self.scale
        return dense

    @tf.function
    def d2mask(self, x):
        mask = self.unet_mask(x / self.scale)
        return mask

    def get_config(self):
        config = {"scale": self.scale}
        config.update(self.kwargs)
        return config
