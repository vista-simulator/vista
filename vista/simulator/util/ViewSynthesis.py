import tensorflow as tf
import numpy as np
import scipy.interpolate
import os
import sys
import time
from enum import Enum

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.client import device_lib

from . import Camera

MAX_DIST = 10000.
HAS_GPU = len([x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']) > 0

class DepthModes(Enum):
    FIXED_PLANE = 1
    INPUT_DISP = 2
    MONODEPTH = 3

class ViewSynthesis:
    """Object to create rotated frames.
    This object is initialized in Trace.py, then called in step(). The step
    function calls .process(R,trans), which handles the frame manipulation based
    on the input rotation and translation matrices. show_plots() is called within
    process().

    Args:
        camera (obj): The camera object to use.

    Attributes:
        cpu_coords (arr): Matrix of x, y, z coordinates held in CPU
        cpu_coords_transposed (arr): Transposed CPU coords to save time in process()
        sess (TF): The current TensorFlow session
        camera (obj): The camera object to use
        _K (arr): The intrinsic camera matrix
        _K_inv (arr): The inverse of the intrinsic camera matrix
        inds (arr): Indices to extract from the data to speed up computation time
        new_image_coords (arr): Contains the real-world location of pixel data
        depth (arr): Array holding real-world depth of each pixel; initialized as all 0s
     """
    def __init__(self,
                camera_obj, # Camera object of the images
                sess=None, # tf.Session object if already defined
                baseline = 0.42567, # [m]
                lookahead_distance=20.0, # lookahead distance for recovery
                mode=DepthModes.FIXED_PLANE,
                monodepth_ckpt=None
        ):
        self.camera = camera_obj
        self.baseline = baseline
        self.lookahead_distance = lookahead_distance
        self.mode = mode
        self.monodepth_ckpt = monodepth_ckpt
        if sess == None:
            self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True))
        else:
            self.sess = sess

        cam_w = self.camera.get_width()
        cam_h = self.camera.get_height()

        xx, yy = np.meshgrid( np.arange(cam_w), np.arange(cam_h) )
        xx = xx.reshape((-1))
        yy = yy.reshape((-1))
        zz = np.ones(xx.size)
        cpu_coords_transposed = np.stack((xx,yy,zz), axis=0)

        img_coords_ph = tf.placeholder(tf.float32, (3, cam_w*cam_h), name='coords_T')
        self.img_coords = tf.get_variable(name='image_coords_transposed', initializer=img_coords_ph, trainable=False, dtype=tf.float32)
        self.sess.run(self.img_coords.initializer, feed_dict={img_coords_ph: cpu_coords_transposed})

    def get_as_tf_op(self):
        """
        TODO: docstring...
        """
        def view_synthesizer_op(theta,
                                translation_x,
                                translation_y,
                                img,
                                disparity=None,
                                curvature=tf.placeholder(tf.float32) ):

            cam_w = self.camera.get_width()
            cam_h = self.camera.get_height()

            with tf.device("/{}:0".format("gpu" if HAS_GPU else "cpu")):
                img_batch = tf.expand_dims(img, 0, name='img_batch')

                # Compute new image
                K = tf.constant(self.camera.get_K(), name='K', dtype=tf.float32)
                K_inv = tf.constant(self.camera.get_K_inv(), name='K_inv', dtype=tf.float32)
                normal = tf.reshape(tf.constant(self.camera.get_ground_plane()[0:3], name='normal'), [1,3])
                d = tf.constant(self.camera.get_ground_plane()[3], name='d', dtype=tf.float32)

                world_coords = tf.matmul(K_inv, self.img_coords, name='2d_to_3d')

                if self.mode == DepthModes.FIXED_PLANE:
                    k = tf.divide(d, tf.matmul(normal, world_coords))
                    depth = tf.where(k < 0, tf.ones(tf.shape(k))*MAX_DIST, k, name="depth")

                elif self.mode == DepthModes.INPUT_DISP:
                    depth_img = tf.clip_by_value(self.baseline * K[0,0] / (disparity*cam_w), 0, MAX_DIST)
                    depth = tf.reshape(depth_img, [1,-1], name='depth')

                elif self.mode == DepthModes.MONODEPTH:
                    train_saver = tf.train.import_meta_graph(self.monodepth_ckpt+'.meta', input_map={"left_input:0": img_batch})
                    train_saver.restore(self.sess, self.monodepth_ckpt)

                    disparity = tf.get_default_graph().get_tensor_by_name("disparities/disp_left_full:0")
                    depth_img = tf.clip_by_value(self.baseline * K[0,0] / (disparity*cam_w), 0, MAX_DIST)
                    depth = tf.reshape(depth_img, [1,-1], name='depth')

                point_on_plane = tf.multiply(depth, world_coords)

                R = tf.stack([[tf.cos(theta), 0.0, -tf.sin(theta)],
                    [0.0, 1.0, 0.0],
                    [tf.sin(theta), 0.0, tf.cos(theta)]])
                t = tf.stack([[translation_x], [0], [translation_y]])

                transformed_coords = tf.matmul(R, point_on_plane) + t
                new_img_coords = tf.matmul(K, transformed_coords)
                new_img_coords = tf.divide(new_img_coords, new_img_coords[2])

                query_pts = tf.transpose(tf.reverse(new_img_coords[0:2,:], axis=[0]))


                # FIXME: CREATE SECOND POINTCLOUD WITH -t_y, TO FIX MASK BUG ON Y-AXIS
                R2 = tf.stack([[tf.cos(theta), 0.0, -tf.sin(theta)],
                    [0.0, 1.0, 0.0],
                    [tf.sin(theta), 0.0, tf.cos(theta)]])
                t2 = tf.stack([[translation_x], [0], [-translation_y]])
                transformed_coords2 = tf.matmul(R2, point_on_plane) + t2
                new_img_coords2 = tf.matmul(K, transformed_coords2)
                new_img_coords2 = tf.divide(new_img_coords2, new_img_coords2[2])
                query_pts2 = tf.transpose(tf.reverse(new_img_coords2[0:2,:], axis=[0]))

                flattened_query_pts = tf.reshape(query_pts, [1, cam_h*cam_w, 2])

                new_image = self.__interpolate_bilinear_new(img_batch, flattened_query_pts)
                new_image = tf.reshape(new_image, [cam_h, cam_w, 3], name="new_image")

                depth_map = tf.reshape(depth, [cam_h, cam_w, 1])
                disp_map = self.baseline * K[0,0] / depth_map

                mask = tf.scatter_nd(tf.cast(query_pts2, tf.int32), tf.ones(cam_h*cam_w), shape=(cam_h, cam_w))
                mask = tf.reshape(mask, (1, cam_h, cam_w, 1))
                mask = tf.nn.max_pool(mask, (1,2,2,1), (1,2,2,1), padding='SAME')
                mask = tf.reverse( tf.reshape(mask, (cam_h//2, cam_w//2)), axis=[1])
                mask = tf.cast(tf.cast(mask, tf.bool), tf.uint8)
                # mask = tf.cast( tf.reverse( tf.reshape(mask, (cam_h/2, cam_w/2)), axis=[1])>0, tf.bool)

                # Compute new curvature #FIXME: does not take into account translation_y
                correction_rotation = theta / self.lookahead_distance
                correction_translation = -2.0 * translation_x / (tf.square(self.lookahead_distance) + tf.square(translation_x))
                new_curvature = curvature - correction_translation - correction_rotation

            return new_image, new_curvature, disp_map, mask

        return view_synthesizer_op


    def get_as_py_func(self):
        """
        TODO: docstring...
        """

        with tf.device("/{}:0".format("gpu" if HAS_GPU else "cpu")):
            theta_ph = tf.placeholder(tf.float32, name='R')
            translation_x_ph = tf.placeholder(tf.float32, name='t_x')
            translation_y_ph = tf.placeholder(tf.float32, name='t_y')
            image_ph = tf.placeholder(tf.float32, (self.camera.get_height(), self.camera.get_width(), 3))
            disp_ph = tf.placeholder(tf.float32, (self.camera.get_height(), self.camera.get_width(), 1))
            curvature_ph = tf.placeholder(tf.float32)

            view_synthesizer_op = self.get_as_tf_op()
            new_image, new_curvature, depth_map, mask = view_synthesizer_op(theta_ph, translation_x_ph, translation_y_ph, image_ph, disp_ph, curvature_ph)

        def func(theta, translation_x, translation_y, image=None, disp=None, curvature=None, return_depth=False, return_mask=False):
            feed_dict = {theta_ph: theta, translation_x_ph: translation_x, translation_y_ph: translation_y}
            outputs = []
            if image is not None:
                feed_dict[image_ph] = image
                outputs.append(new_image)
            if disp is not None:
                feed_dict[disp_ph] = disp
            if curvature is not None:
                feed_dict[curvature_ph] = curvature
                outputs.append(new_curvature)
            if return_depth:
                outputs.append(depth_map)
            if return_mask:
                outputs.append(mask)

            return self.sess.run(outputs, feed_dict=feed_dict)

        return func

    def __interpolate_bilinear_new(self, grid, query_points):
        """
        TODO: docstring.. can probably copy a lot from the official tensorflow docstring here...
        """

        # NOTE:  GPU gather op requires gpu indices to be in host memory.
        # When you int32 indicies, they get placed on GPU, and then
        # executing the op requires the transfer of indices from GPU to CPU.
        # We incure an additional transfer penalty after doing the gather
        # op since we have to move the data back from CPU to GPU.
        #
        # We can solve this problem by defining the indicies as int64 (not
        # int32), thus forcing them to live on the CPU and eliminating the
        # forced transfers!
        # see: https://stackoverflow.com/questions/43816698/tensorflow-difference-in-performance-of-gather-for-int32-and-int64-indices-on-g

        with ops.name_scope('interpolate_bilinear'):

            grid = ops.convert_to_tensor(grid)
            query_points = ops.convert_to_tensor(query_points)
            shape = grid.shape
            if len(shape) != 4:
                msg = 'Grid must be 4 dimensional. Received size: '
                raise ValueError(msg + str(grid.get_shape()))

            batch_size, height, width, channels = shape
            query_type = query_points.dtype
            grid_type = grid.dtype

            if (len(query_points.get_shape()) != 3 or query_points.get_shape()[2].value != 2):
                msg = ('Query points must be 3 dimensional and size 2 in dim 2. Received size: ')
                raise ValueError(msg + str(query_points.get_shape()))


            # flow_dist = tf.norm(flow, axis=2)
            # easy_query_points = tf.less(flow_dist, 1.0)
            # interp_full = tf.where(easy_query_points[0], query_points[0], tf.zeros_like(query_points[0]))
            #
            # # import pdb; pdb.set_trace()
            # # which_easy = tf.boolean_mask([tf.range(query_points.shape[1])], easy_query_points)
            # which_hard = tf.boolean_mask([tf.range(query_points.shape[1])], ~easy_query_points)
            # import pdb; pdb.set_trace()
            # query_points = tf.expand_dims(tf.boolean_mask(query_points, ~easy_query_points, axis=0), axis=0)
            #
            # _, n_queries, _ = query_points.shape


            unstacked_query_points = array_ops.unstack(query_points, axis=2)

            alphas = []
            floors = []
            ceils = []

            for dim in [0,1]:
                queries = unstacked_query_points[dim]
                size_in_dimension = shape[dim + 1]

                # max_floor is size_in_indexing_dimension - 2 so that max_floor + 1
                # is still a valid index into the grid.
                max_floor = math_ops.cast(size_in_dimension - 2, query_type)
                min_floor = constant_op.constant(0.0, dtype=query_type)
                floor = math_ops.minimum(
                    math_ops.maximum(min_floor, math_ops.floor(queries)), max_floor)
                int_floor = math_ops.cast(floor, dtypes.int64)
                floors.append(int_floor)

                int_ceil = int_floor + 1
                ceils.append(int_ceil)

                # alpha has the same type as the grid, as we will directly use alpha
                # when taking linear combinations of pixel values from the image.
                alpha = math_ops.cast(queries - floor, grid_type)
                min_alpha = constant_op.constant(0.0, dtype=grid_type)
                max_alpha = constant_op.constant(1.0, dtype=grid_type)

                # Expand alpha to [b, n, 1] so we can use broadcasting
                # (since the alpha values don't depend on the channel).
                alpha = array_ops.expand_dims(alpha, 2)
                alphas.append(alpha)

            flattened_grid = array_ops.reshape(grid, [batch_size * height * width, channels])
            batch_offsets = array_ops.reshape(math_ops.range(batch_size, dtype=tf.int64) * height * width, [batch_size, 1])

            # This wraps array_ops.gather. We reshape the image data such that the
            # batch, y, and x coordinates are pulled into the first dimension.
            # Then we gather. Finally, we reshape the output back.
            #
            # I also tried using tf.gather_nd instead of tf.gather to make this
            # code cleaner but the runtime is roughly equivalents so I left it
            # with tf.gather.
            def gather(y_coords, x_coords, name):
                with ops.name_scope('gather-' + name):
                    linear_coordinates = batch_offsets + y_coords * width + x_coords
                    gathered_values = array_ops.gather(flattened_grid, linear_coordinates)
                    return array_ops.reshape(gathered_values, [batch_size, -1, channels])

            # grab the pixel values in the 4 corners around each query point
            top_left = gather(floors[0], floors[1], 'top_left')
            top_right = gather(floors[0], ceils[1], 'top_right')
            bottom_left = gather(ceils[0], floors[1], 'bottom_left')
            bottom_right = gather(ceils[0], ceils[1], 'bottom_right')

            # now, do the actual interpolation
            with ops.name_scope('interpolate'):
                interp_top = alphas[1] * (top_right - top_left) + top_left
                interp_bottom = alphas[1] * (bottom_right - bottom_left) + bottom_left
                interp = alphas[0] * (interp_bottom - interp_top) + interp_top

            # interp_full = tf.scatter_update(interp_full, which_hard, interp)

            return interp
