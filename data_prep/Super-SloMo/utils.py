"""
A collection of utilities for searching image/ videos files.
"""
from __future__ import print_function
from __future__ import division

import os
import glob
import numpy as np
import cv2

VIDEO_EXTENSIONS = [".mp4", ".mov", ".m4v", ".avi"]
IMAGE_EXTENSIONS = [".jpg", ".png"]


def is_image(path):
    """Checks if a path is an image
    Args:
        path: file path
    Returns:
        is_image (bool): True or False
    """
    return os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS


def is_video(path):
    """Checks if a path is a video
    Args:
        path: file path
    Returns:
        is_video (bool): True or False
    """
    return os.path.splitext(path)[1].lower() in VIDEO_EXTENSIONS


def grab_images_and_videos(adir):
    """Grabs image and video files
    Args:
        adir: directory with images
    Returns:
        files: image and video files
    """
    return grab_images(adir) + grab_videos(adir)


def grab_images(adir):
    """Grabs image files
    Args:
        adir: directory with images
    Returns:
        files: image files
    """
    return grab_files(adir, [".jpg", ".png"])


def grab_videos(adir):
    """Grabs videos in a directory

    Args:
        adir: directory or list of directory

    Returns:
        files: files with image/ video extension
    """
    files = grab_files(adir, VIDEO_EXTENSIONS)
    # detect jpg directories
    others = os.listdir(adir)
    for sdir in others:
        subdir = os.path.join(adir, sdir)
        imgs = grab_images(subdir)
        if len(imgs):
            # detect number of digits
            img_name = os.path.basename(imgs[0])
            ext = os.path.splitext(img_name)[1]
            nd = sum(c.isdigit() for c in img_name)
            redir = os.path.join(subdir, "%" + str(nd) + "d" + ext)
            files.append(redir)

    return files


def grab_files(adir, extensions):
    """Grabs files with allowed extensions
    Args:
        adir: directory
        extensions: allowed extensions
    Returns:
        files
    """
    all_files = []
    for ext in extensions:
        all_files += glob.glob(adir + "/*" + ext)
        all_files += glob.glob(adir + "/*" + ext.upper())
    return all_files


def convert_to_gray(img):
    """"Converts the input RGB uint8 image to gray scale.

    Args:
        img (np.ndarray): RGB uint8 image
    """
    # img should be in graylevel with three channels
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.dstack([img_gray, img_gray, img_gray])
    return img


def draw_arrows(img, flow, step=16, flow_unit="relative",  convert_img_to_gray=True):
    """
    Visualizes Flow, by drawing hsv-colored arrows on top of the input image.

    Args:
        img (np.ndarray): RGB uint8 image, to draw the arrows on.
        flow (np.ndarray): of shape (2, height width), where the first index is the x component of 
                           the flow and the second is the y-component.
        step (int): Draws every `step` arrow. use to increase clarity, especially with fast motions.
        flow_unit (string): either in "relative" mode, in which case the unit is width / 2 pixels
                            (respectively height / 2), or in absolute "pixels" mode in which case 
                            unit is simply in pixels.
    """
    assert flow_unit in ("relative", "pixels"), "flow_unit should be relative or pixels"
    flow_y = flow[1, ::step, ::step]
    flow_x = flow[0, ::step, ::step]
    c, height, width = flow.shape
    if convert_img_to_gray:
        img = convert_to_gray(img)

    mag, ang = cv2.cartToPolar(flow_x, flow_y, angleInDegrees=True)
    ang /= 2

    hsvImg = np.ones((height // step, width // step, 3), dtype=np.uint8) * 255
    hsvImg[..., 0] = ang
    rgbImg = cv2.cvtColor(hsvImg, cv2.COLOR_HSV2RGB).astype('int')

    ratio = float(img.shape[0]) / height

    x = np.arange(0, width, step)
    y = np.arange(0, height, step)
    x_array, y_array = np.meshgrid(x, y)

    # arrow displacement
    dx = (flow_x * width / 2) if flow_unit == "relative" else flow_x
    dy = (flow_y * height / 2) if flow_unit == "relative" else flow_y

    # computes arrows ending point
    p2x = ((x_array + dx) * ratio).astype("int")
    p2y = ((y_array + dy) * ratio).astype("int")

    x_array = (x_array * ratio).astype("int")
    y_array = (y_array * ratio).astype("int")

    # displaying arrows
    for i in range(0, height // step):
        for j in range(0, width // step):

            color_list = rgbImg[i, j, :].tolist()
            img = cv2.arrowedLine(img, (x_array[i, j], y_array[i, j]), (p2x[i, j], p2y[i, j]), color_list, 1)

    return img
