'''
use this file for all drawing functions specific to the simulator
previously we mentioned putting this in deepknight/util but I think it
makes more sense here (inside the simulator package)
'''
import cv2
from enum import Enum
import math
import numpy as np
import os
import pdb
import sys

deepknight_root = os.environ.get('DEEPKNIGHT_ROOT')
sys.path.insert(0, os.path.join(deepknight_root, 'simulator'))
import assets
assets_path = assets.path

sys.path.insert(0, os.path.join(deepknight_root, 'util/'))
import Image


wheelImg = cv2.resize(cv2.imread(os.path.join(assets_path,'img/mit.jpg')), (360,360))
carImg = cv2.resize(cv2.imread(os.path.join(assets_path,'img/lambo_square.jpg')),(140,300))
roadImg = cv2.resize(cv2.imread(os.path.join(assets_path,'img/road.jpg')), (500, 300))
speedImg = cv2.resize(cv2.imread(os.path.join(assets_path,'img/speed.jpg')), (800, 400))

'''
Enumeration of every module that could be drawn in the window.
To add a new module, add its name to this class and add the associated drawing
function below.
'''
class Module(Enum):
    STEERING = 1
    LANE = 2
    INFO = 3
    FRAME = 4
    SPEED = 5

'''
Class for handling boxes. To initialize, pass in the top left and
bottom right coordinates of the box.
'''
class Box:
    def __init__(self, top_left, bottom_right):
        self.x1 = top_left[0]
        self.y1 = top_left[1]
        self.x2 = bottom_right[0]
        self.y2 = bottom_right[1]

    '''
    scale the box by a tuple factor (scale_x, scale_y)
        @factor: tuple of linear factors to multiply the box dimensions by
    '''
    def scale(self, factor):
        new_top_left     =  (self.x1*factor[0], self.y1*factor[1])
        new_bottom_right =  (self.x2*factor[0], self.y2*factor[1])
        return Box(new_top_left, new_bottom_right) #TODO: might be better to just modify the existing self instead of returning a new Box

    '''
    Cast the box cordinates to integers.
    '''
    def round(self):
        self.x1 = int(self.x1)
        self.x2 = int(self.x2)
        self.y1 = int(self.y1)
        self.y2 = int(self.y2)

    '''
    Return the width of the box
    '''
    def get_width(self):
        return self.x2 - self.x1

    '''
    Return the height of the box
    '''
    def get_height(self):
        return self.y2 - self.y1

    def __getitem__(self, key): #this is only so previous code does not break... TODO: will remove once old code changes to start using boxes like `draw_steering_wheel`.
        return [(self.x1, self.y1), (self.x2, self.y2)][key]

def normal_resize(img, sz):
    return cv2.resize(img, sz)

'''
Resize an image to a given size but maintain its original aspect ratio.
If the aspect ratio of the original image is different than the aspect
ratio of the final shape, then resize it to fit as tight as possible within
the new size.
    @img: image to resize
    @sz: (width, height) of desired output shape -- NOTE: follows openCV convention
'''
def resize_keep_ratio(img, sz):

    desired_ratio = float(img.shape[1]) / img.shape[0]
    box_ratio = float(sz[0]) / sz[1]

    if desired_ratio > box_ratio:
        # we are bounded by the numerator (width) of the desired ratio
        width = sz[0]
        height = int(width / desired_ratio)
        bounded_resize = normal_resize(img, (width, height))
        pad_top = int( (sz[1] - height)/2. )
        pad_bottom = sz[1] - height - pad_top

        return cv2.copyMakeBorder(bounded_resize, top=pad_top, bottom=pad_bottom, right=0, left=0, borderType=cv2.BORDER_CONSTANT,value=[0,0,0])

    elif desired_ratio < box_ratio:
        # we are bounded by the denominator (height) of the desired ratio
        height = sz[1]
        width = int(height * desired_ratio)
        bounded_resize = normal_resize(img, (width, height))
        pad_right = int( (sz[0] - width)/2. )
        pad_left = sz[0] - width - pad_right

        return cv2.copyMakeBorder(bounded_resize, top=0, bottom=0, right=pad_right, left=pad_left, borderType=cv2.BORDER_CONSTANT,value=[0,0,0])

    else: #desired_ratio == box_ratio
        return normal_resize(img, sz)

#each function is its own module, everything localized within itself
# -- i.e. draw car in middle of road, then put car/road module into window using fuse
def draw_steering_wheel(window_size, box, degrees):
    #change to pixels
    window_box = box.scale(window_size)
    width = int(window_box.get_width())
    height = int(window_box.get_height())

    rows, cols, z = wheelImg.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), degrees, 1)
    image = cv2.warpAffine(wheelImg, M, (cols, rows))
    finalImg = resize_keep_ratio(image, (width, height))
    return finalImg

def draw_mini_car_on_road(window_size, box, translation, car_rotation, road_width, road=False):
    #change to pixels
    window_box = box.scale(window_size)
    window_width = int(window_box.get_width())
    window_height = int(window_box.get_height())

    #setup images
    mini_car_height = int(window_height/2)
    mini_car_width = mini_car_height
    car_resized = cv2.resize(carImg, (mini_car_width, mini_car_height))

    #rotation
    rows, cols, z = car_resized.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), car_rotation*180/math.pi, 1)
    car_rotated = cv2.warpAffine(car_resized,M,(cols, rows))

    #road image work
    if road:
        mini_car_on_road = resize_keep_ratio(roadImg, (window_width, window_height))
        road_shape = mini_car_on_road.shape #because width of image doesn't necessarily equal width of window
        red_zone = (road_shape[1]/10.)/road_shape[1] #adjust here the actually red lane size
        scalar = 1-2.*red_zone-2*(float(mini_car_width)/road_shape[1])-.2 #the extra .2 is because I'm not sure why
        translation = translation*scalar #scalar to stay on picture untill crash condition roughly
        car_center = int(((translation/road_width) + 1/2.) * road_shape[1])
        mini_car_on_road[mini_car_height-mini_car_height/2 : window_height/2-mini_car_height/2+mini_car_height,
                         car_center-mini_car_width/2:car_center-mini_car_width/2+mini_car_width] = car_rotated
    else:
        #draw line representation
        mini_car_on_road = np.zeros((window_height, window_width,3), np.uint8)
        mini_road_width = mini_car_on_road.shape[1]
        mini_lane_space = mini_road_width - mini_car_width/2 #each lane marker is mini_car_width/4 thick
        mini_free_space = mini_lane_space - mini_car_width/2 #reduces free space so that image can translate upto touching lane marker

        relative_car_center = int(((float(translation)/road_width) + .5) * mini_free_space) #localized within mini_free_space
        diff = window_width/2 - mini_free_space/2 #need to add to get mini_free_space module to center of mini_car_on_road module
        car_center = relative_car_center + diff #localizes mini_free_space within mini car module

        #handle crash events that have greater translations than allowed
        if car_center > mini_free_space:
            car_center = mini_free_space + mini_car_width/2.5
        if car_center < mini_car_width:
            car_center = mini_car_width/1.5

        y_top = int(window_height/2-mini_car_height/2+mini_car_height)
        y_bot = int(window_height/2-mini_car_height/2)

        x_r = int(car_center-mini_car_width/2+mini_car_width)
        x_l = int(car_center-mini_car_width/2)

        adjust_y = (y_top - y_bot) - car_rotated.shape[1]
        adjust_x = ( x_r - x_l) - car_rotated.shape[0]

        mini_car_on_road[y_bot : y_top - adjust_y, x_l : x_r - adjust_x] = car_rotated

        #apply lines -- iterate to make dashed
        stripe_length = int(window_height/6)
        stripe_gap = 2*stripe_length
        for i in range(0,window_height, stripe_gap):
            cv2.rectangle(mini_car_on_road, (mini_car_width/8, i),(mini_car_width/4, i+stripe_length),(255,255,255), -1)
        for i in range(0,window_height, stripe_gap):
            cv2.rectangle(mini_car_on_road, (window_width - mini_car_width/8, i),(window_width-mini_car_width/4, i+stripe_length),(255,255,255), -1)

    return mini_car_on_road

def draw_info_sidebar(window_size, box, info):
    window_box = box.scale(window_size)
    width = int(window_box.get_width())
    height = int(window_box.get_height())

    info_sidebar = np.zeros((height, width, 3), np.uint8)
    #extract info from dict
    elapsed_time = "Elapsed Time: " + str(round(info['timestamp'] - info['first_time'], 1)) + " sec"
    current_time = "Current Frame: " #+ str(self.current_frame_index)
    steering_angle = "Steering Angle: " + str(round(info['model_angle'], 1)) + " deg."
    actual_curv = "Human Curvature: " + str(round(info['human_curvature'], 6))
    predicted_curv = "Model Curvature: " + str(round(info['model_curvature'], 6)) # TODO be careful if its adjusted to the human
    distance = "Distance: " + str(round(info['distance'],2)) + ' m'
    velocity = "Velocity: " + str(round(2.23*info['model_velocity'], 1)) + " mph"
    translation = "Translation: " + str(round(info['translation'],6)) + " m"
    rotation = "Rotation: " + str(round(info['rotation'],6)) + " rad"

    #apply text to array
    cv2.putText(info_sidebar, elapsed_time, (int(width/20),int(height/10)), cv2.FONT_HERSHEY_SIMPLEX, width/500., (255,255,255), 1)
    cv2.putText(info_sidebar, current_time, (int(width/20),int(2*height/10)), cv2.FONT_HERSHEY_SIMPLEX, width/500., (255,255,255), 1)
    cv2.putText(info_sidebar, steering_angle, (int(width/20),int(3*height/10)), cv2.FONT_HERSHEY_SIMPLEX,width/500. , (255,255,255), 1)
    cv2.putText(info_sidebar, actual_curv, (int(width/20),int(4*height/10)), cv2.FONT_HERSHEY_SIMPLEX, width/500., (255,255,255), 1)
    cv2.putText(info_sidebar, predicted_curv, (int(width/20),int(5*height/10)), cv2.FONT_HERSHEY_SIMPLEX, width/500., (255,255,255), 1)
    cv2.putText(info_sidebar, distance, (int(width/20),int(6*height/10)), cv2.FONT_HERSHEY_SIMPLEX, width/500., (255,255,255), 1)
    cv2.putText(info_sidebar, velocity, (int(width/20),int(7*height/10)), cv2.FONT_HERSHEY_SIMPLEX, width/500., (255,255,255), 1)
    cv2.putText(info_sidebar, translation, (int(width/20),int(8*height/10)), cv2.FONT_HERSHEY_SIMPLEX, width/500., (255,255,255), 1)
    cv2.putText(info_sidebar, rotation, (int(width/20),int(9*height/10)), cv2.FONT_HERSHEY_SIMPLEX, width/500., (255,255,255), 1)
    final = resize_keep_ratio(info_sidebar, (width, height))

    return final

def draw_speedometer(window_size, box, human=None, model=None):
    #change to pixels
    window_box = box.scale(window_size)
    width = int(window_box.get_width())
    height = int(window_box.get_height())
    image = speedImg.copy()

    max_speed = 100.
    R = int(image.shape[0]*0.8)
    origin = (image.shape[1]//2, image.shape[0])
    color = [(0,0,255), (255,0,0)]
    speeds = [human, model]
    for i, speed in enumerate(speeds):
        if speed is None: continue
        angle = -np.pi * (speed/max_speed + 1)
        end = (origin[0]+int(R*np.cos(angle)), origin[1]-int(R*np.sin(angle)))
        cv2.line(image, tuple(origin), tuple(end), color[i], image.shape[0]//80)

    finalImg = resize_keep_ratio(image, (width, height))
    return finalImg


def draw_frame(window_size, box, frame, camera, model_projection, true_projection):
    window_box = box.scale(window_size)
    width = int(window_box.get_width())
    height = int(window_box.get_height())

    model_T = np.int32([ Image.draw_noodle(model_projection, camera).T ])
    true_T = np.int32([ Image.draw_noodle(true_projection, camera).T ])

    cv2.polylines(frame, model_T, False, (0,0, 255), int(width/25.))
    cv2.polylines(frame, true_T, False, (255,0,0), int(width/25.))

    frame = Image.draw_box_new(frame, camera.get_roi(), linewidth=int(width/90.))

    uncropped_height, uncropped_width = frame.shape[:2]
    new_frame = Image.draw_car_hood(frame)
    new_frame = new_frame[int(0.2*uncropped_height):-int(0.2*uncropped_height), int(0.2*uncropped_width):-int(0.2*uncropped_width) ] #this crops only the relevant part of the frame

    cv2.resize(new_frame, (width, height))
    finalImg = resize_keep_ratio(new_frame, (width, height))
    return finalImg

def fuse(window_size, modules_list, coordinates_list, draw_outline=True): #order of modules: steering wheel, road, info sidebar, frame #window_size passed in as X Y Z
    window = np.zeros(((window_size[1], window_size[0], window_size[2])), np.uint8)
    for i in range(len(modules_list)):
        top_left = int(coordinates_list[i][0][0]*window_size[0]), int(coordinates_list[i][0][1]*window_size[1])
        bot_right = int(coordinates_list[i][1][0]*window_size[0]), int(coordinates_list[i][1][1]*window_size[1])
        adjust_x = 1 if modules_list[i].shape[1] != bot_right[0] - top_left[0] else 0
        adjust_y = 1 if modules_list[i].shape[0] != bot_right[1] - top_left[1] else 0
        window[top_left[1]:bot_right[1]-adjust_y,top_left[0]:bot_right[0]-adjust_x] = modules_list[i]
        if draw_outline:
            #draw rectange over the module for debugging
            box = Box(coordinates_list[i][0], coordinates_list[i][1]) #TODO dont do this! just pass in the dictionary of boxes instead of coordinates_list (see steering module example)
            window_box = box.scale(window_size)
            window_box.round()
            cv2.rectangle(window,window_box[0],window_box[1],(255,0,0),3)
    return window


#modules_list = [steering_wheel, mini_car_on_road, info_mod, frame]
#coordinates_list = [(375, 875),(775,875),(875, 250), (400, 375)]

# etc...
# # #tests

# #info test
# info = {'model_angle': 0.0, 'timestamp': 1512238554.630844, 'prev_velocity': np.array(7.2316927), 'curvature': 0.00012857774529911864, 'first_time': 1512238550.331, 'prev_curvature': -0.00012857774529911864}
# info_mod = draw_info_sidebar(.5, info)
# # cv2.imshow('info bar test', info_mod)
# # cv2.namedWindow('info bar test', cv2.WINDOW_NORMAL)
# # cv2.waitKey(2000)

# #steering wheel test
# steering_wheelImg = cv2.imread(assets_path + '/img/mit.jpg')
# steering_wheel = draw_steering_wheel(.2, steering_wheelImg, 2 ) #input angle in rad
# #cv2.imshow('steering wheel test', steering_wheel)
# #cv2.namedWindow('steering wheel test', cv2.WINDOW_NORMAL)
# #cv2.waitKey(1)
#
# #mini road test
# carImg = cv2.imread(assets_path + '/img/lambo.jpg')
# roadImg = cv2.imread(assets_path + '/img/road2.jpg')
# mini_car_on_road = draw_mini_car_on_road_new((500,500),Box((0,0),(1,1)), carImg, roadImg, 0, 1)
# cv2.imshow('Mini car on road test',mini_car_on_road)
# cv2.namedWindow('Mini car on road test',cv2.WINDOW_NORMAL)
# cv2.waitKey(1)
#
# #frame test
# frameImg = cv2.imread(assets_path + '/img/example_frame.jpg')
# sizedImg = cv2.resize(frameImg, (int(1920/1.58940397351), 1920)) #inputs as size of actual trace frame
# frame = draw_frame(.3, sizedImg)
# cv2.imshow('frame test', frame)
# cv2.namedWindow('frame test', cv2.WINDOW_NORMAL)
# cv2.waitKey()
#
# # speedometer test
# out = draw_speedometer((500,500), Box((0,0),(1,1)), human=10, model=20)
# cv2.imshow('speed', out)
# cv2.waitKey()
#
#
# #fuse test
# modules_list = [steering_wheel, mini_car_on_road, info_mod, frame]
# coordinates_list = [(375, 875),(775,875),(875, 250), (400, 375)]
# window = fuse(modules_list, coordinates_list)
# cv2.imshow('fuse test', window)
# cv2.namedWindow('fuse test', cv2.WINDOW_NORMAL)
# cv2.waitKey()
