'''
Collection of functions that can be done on an images, some require
Camera object to be passed in as well.
'''
import cv2
import collections
import copy
import os
import sys
import math
import numpy as np
try:
    import importlib.resources as pkg_resources
except ImportError: # py < 3.7
    import importlib_resources as pkg_resources

from . import Camera

resources_root = pkg_resources.path("vista.resources", "")
with resources_root as rr:
    roadImg = cv2.resize(cv2.imread(os.path.join(rr,'img/road.jpg')), (500, 300))
    car_hood_path = os.path.join(rr, 'img/camera_front_hood_black.png')
    car_hood = cv2.imread(car_hood_path, cv2.IMREAD_UNCHANGED)

car_hood_cache = collections.OrderedDict()

#FIXME change positions of box from bottom left -> topleft and topright to bottom right
def crop(image, box, batch=False):
    """ Crop image to specified bounding box

    Args:
        image (np.array): image to crop
        box (list): the four coordinates defining the box
            box = [i1,j1,i2,j2]
            i1,j1-------
            |          |
            |          |
            |          |
            -------i2,j2
        angle (float): angle of the roi within the bounding box
        batch (bool): if the first dimension is a batch dimension or not

    Returns:
        new_img (np.array): the cropped image

    """
    (i1,j1,i2,j2) = box
    if batch:
        return image[:, i1:i2, j1:j2]
    else:
        return image[i1:i2, j1:j2]

def crop_center(image, shape, batch=False):
    """
    TODO: docstring.
    """
    old_shape = image.shape
    h,w = old_shape[1:-1] if batch else old_shape[:-1]
    hpad = (w - shape[1])//2
    vpad = (h - shape[0])//2
    return image[:, vpad:-vpad, hpad:-hpad] if batch else image[vpad:-vpad, hpad:-hpad]

def rotate(image, angle, batch=False):
    """
    TODO: docstring.
    """
    image_center = tuple(np.array(image.shape[1::-1]) / 2.)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle*180./np.pi, 1.0)
    return cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.WARP_INVERSE_MAP)

def crop_rotated(image, box, angle, new_size, batch=False):
    """
    TODO: docstring.
    """
    cropped = crop(image, box, batch)
    rotated = rotate(cropped, angle, batch)
    return crop_center(rotated, new_size)

def draw_box(image, box):
    """ Draw box over an image.

    TODO: docstring.

    """
    (h,w) = image.shape[:2]
    (x1,y1,x2,y2) = [int(round(box[0]*w)), int(round(box[1]*h)),
                     int(round(box[2]*w)), int(round(box[3]*h))]

    return cv2.rectangle(image,(x1,y2),(x2,y1),(0,255,0),3)

def draw_box_new(image, box, linewidth=1):
    if len(box)==4 and isinstance(box[0], int):
        (i1,j1,i2,j2) = box
        cv2.rectangle(image,(j1,i1),(j2,i2),(0,255,0),linewidth)
    else:
        cv2.polylines(image, box, True, (0,255,0), linewidth)

    return image

def draw_noodle(curvature, camera, max_dist=30):
    """
    TODO: docstring.
    """

    K = camera.get_K()
    normal = camera.get_ground_plane()[0:3]
    normal = np.reshape(normal, [1,3])
    d = camera.get_ground_plane()[3]
    A, B, C = normal[0]

    max_turning_radius = 3.5 #FIXME dont hardcode since its re-used this value in a couple other places (ex. Trace)
    curvature = np.clip(curvature, -1/max_turning_radius, 1/max_turning_radius)
    radius = 1. / (curvature + 1e-9) # to avoid divide by zero; negate to fit right hand rule

    z_vals = np.linspace(0, np.clip(abs(radius), 3, max_dist), 20) #future values of the curve to find noodle points for
    y_vals = (d - C*z_vals)/B #assumes A is zero
    x_sq_r = radius**2 - z_vals**2 - (y_vals-d)**2
    x_vals = np.sqrt(x_sq_r[x_sq_r > 0]) - abs(radius)
    y_vals = y_vals[x_sq_r > 0]
    z_vals = z_vals[x_sq_r > 0]

    if radius < 0:
        x_vals *= -1

    world_coords = np.stack((x_vals, y_vals, z_vals)) #3 col, 20 row

    theta = camera.get_yaw()
    R = np.array([[np.cos(theta), 0.0, -np.sin(theta)],[0.0, 1.0, 0.0],[np.sin(theta), 0.0, np.cos(theta)]])
    tf_world_coords = np.matmul(R, world_coords)
    img_coords = np.matmul(K,tf_world_coords)
    norm = np.divide(img_coords, img_coords[2]+1e-10)

    valid_inds = np.multiply(norm[0] >= 0 , norm[0] < camera.get_width())
    valid_inds = np.multiply(valid_inds, norm[1] >= 0)
    valid_inds = np.multiply(valid_inds, norm[1] < camera.get_height())

    image_coords = norm[:2, valid_inds].astype(int)

    return image_coords

def draw_noodle_cartesian(camera, x, y, z=None):
    """
    TODO: docstring.
    """

    K = camera.get_K()
    A,B,C,d = camera.get_ground_plane()
    q = copy.copy(camera._quaternion)
    q_ = copy.copy(camera._quaternion); q_[1:] *= -1
    # print q, q_

    def quaternion_multiply(quaternion1, quaternion0):
        x0, y0, z0, w0 = quaternion0
        x1, y1, z1, w1 = quaternion1
        return np.array([x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                         -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                         x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
                         -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0], dtype=np.float64)


    # y_vals = (d - C*z_vals)/B #assumes A is zero
    # x_vals =  np.sqrt(radius**2 - z_vals**2 - (y_vals-d)**2) - abs(radius)
    #
    # x_vals[radius < 0] *= -1

    # print q, q_

    world_coords = np.stack((x, 1.6-y, z)) #3 col, 20 row
    # world_coords_in_cam = np.zeros_like(world_coords)
    # for i in range(world_coords.shape[1]):
    #     p = world_coords[:,i]
    #     p = np.append(p, 0)
    #     world_coords_in_cam[:,i] = quaternion_multiply(quaternion_multiply(q_, p), q).flatten()[:3]
    #
    # world_coords_in_cam *= -1

    theta_camera = 1.6/180.*np.pi
    R = np.array([[1.0,0.0,0.0], [0.0, np.cos(theta_camera), -np.sin(theta_camera)],[0, np.sin(theta_camera), np.cos(theta_camera)]])
    world_coords_in_cam = np.matmul(R, world_coords)

    # print world_coords
    # print world_coords_in_cam

    theta = camera.get_yaw()
    R = np.array([[np.cos(theta), 0.0, -np.sin(theta)],[0.0, 1.0, 0.0],[np.sin(theta), 0.0, np.cos(theta)]])
    tf_world_coords = np.matmul(R, world_coords_in_cam)
    # print tf_world_coords

    img_coords = np.matmul(K,tf_world_coords)

    norm = np.divide(img_coords, img_coords[2])

    valid_inds = np.multiply(norm[0] >= 0 , norm[0] < camera.get_width())
    valid_inds = np.multiply(valid_inds, norm[1] >= 0)
    valid_inds = np.multiply(valid_inds, norm[1] < camera.get_height())

    image_coords = norm[:2, valid_inds].astype(int)

    # print image_coords

    return image_coords


def draw_car_hood(img):
    """ Draws a car hood over the image frame

    Args:
        img (np.array): image frame taken from a front facing camera

    Returns:
        new_img (np.array): the same image provided as input with the exception
        that the car hood has been placed on the appropriate mask
    """
    if img.shape[:2] not in car_hood_cache:
        # if image of correct size is not in cache add it
        hood, mask = (car_hood[:,:,:3], car_hood[:,:,3])
        new_hood = cv2.resize(hood, img.shape[:2][::-1])
        new_mask = cv2.resize(mask, img.shape[:2][::-1])
        mask_inds = np.nonzero(new_mask.reshape((-1)))
        mask_vals = new_hood.reshape((-1,3))[mask_inds]
        car_hood_cache[img.shape[:2]] = (mask_vals, mask_inds)

    mask_vals, mask_inds = car_hood_cache[img.shape[:2]]
    img_flat = img.reshape((-1,3))
    img_flat[mask_inds] = mask_vals
    img = img_flat.reshape(img.shape[:3])

    return img


################################################################################
################################################################################
############################### LEGACY FUNCTIONS ###############################
################################## DEPRECATED ##################################
################################################################################
################################################################################


''' adjust brightness by factor x '''
def adjust_brightness(image, x):
    pass

''' adjust contrast by factor x '''
def adjust_contrast(image, x):
    pass

''' adjust gamma by factor x '''
def adjust_gamma(image, x):
    pass

def extract_roi(image, camera, fov, lookahead): #(?)#
    pass

def view_to_camera(image, source, target):
    pass


'''
Randomly perturb an image and adjust the control value
'''
def random_pertubation(image, inverse_r):
    NORMAL = np.array([[0, 1, 0.55 ]]) #BAD CODE
    INTRINSIC_MATRIX = np.array([[ 942.21001829,    0.        ,  982.06345972],
              [   0.        ,  942.91483399,  617.04862261],
              [   0.        ,    0.        ,    1.        ]]) #BAD CODE
    camera = Camera(INTRINSIC_MATRIX, NORMAL, image.shape)

    #theta = np.random.normal(scale=0.015) #approx sigma=10 degrees
    translate = np.random.normal(scale=0.3) #approx sigma=0.3 meters

    #c = np.cos(theta)
    #s = np.sin(theta)
    #R = np.array([[c,0,-s], [0,1,0], [s,0,c]])
    t = np.array([translate, 0, 0])

    # rotate first on whole image
    #t_blank = np.array([[0, 0, 0]])
    #rotated_frame = camera.homography(frame, R, t_blank)

    # then translate on bottom half
    R_blank = np.eye(3)
    translated_frame = camera.homography(image, R_blank, t, 0.52)

    lookahead = 5.
    wheel_base = 2.77876
    adjusted_control = inverse_r - translated / lookahead / wheel_base #some function of inverse_r

    return (translated_frame, adjusted_control)


'''
##  Performs a homography on the given frame
##   Inputs:
#       image = an image to perform homography on
#       R = (3x3) rotation matrix
#       t = translation matrix
#       horizon_offset = val in range [0,1] that denotes where the horizon line is in the frame
##   Outputs:
#       stitched = the input image shifted by the specified rotation and translation
'''
def homography(image, camera, R, t, horizon_offset=0):
    # construct homography matrix

    H = R - np.dot(t.T, camera.get_normal())

    IMG_SCALE_Y = camera.get_height() / image.shape[0]
    IMG_SCALE_X = camera.get_width() / image.shape[1]

    IMG_SCALE = np.array([[1./IMG_SCALE_X,           0, 0],
                             [       0, 1./IMG_SCALE_Y, 0],
                             [       0,           0, 1.]])

    scaled_K = np.matmul(IMG_SCALE, camera.get_K())

    KHK = np.matmul(np.matmul(scaled_K, H), np.linalg.inv(scaled_K))

    # crop and perform transformation
    cropped_image = image[int(image.shape[0]*horizon_offset):]
    shifted_crop = cv2.warpPerspective(cropped_image, KHK, (cropped_image.shape[1], cropped_image.shape[0]), flags=cv2.INTER_NEAREST)

    # stich back together
    stitched = np.concatenate((image[:int(image.shape[0]*horizon_offset)], shifted_crop), axis=0)

    return stitched
