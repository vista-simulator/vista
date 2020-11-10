import pygame
import numpy as np


class Joystick:
    ''' Interface to handle joystick communication '''
    def __init__(self, ind=0):
        pygame.init()

        pygame.joystick.init()

        self.joystick = pygame.joystick.Joystick(ind)
        self.joystick.init()

        self.first_throttle = True
        self.first_brake = True

    def __sync(self):
        ''' syncronizes pygame's internal event handler '''
        pygame.event.pump()

    def get_x(self):
        ''' returns the current x coordinate '''
        self.__sync()
        return self.joystick.get_axis(0)

    def get_y(self):
        ''' returns the current y coordinate '''
        self.__sync()
        return -self.joystick.get_axis(1)

    def get_twist(self):
        ''' returns the current twist '''
        self.__sync()
        return -self.joystick.get_axis(2)

    def get_throttle(self):
        ''' returns the current throttle [0,1] '''
        self.__sync()

        alpha = self.joystick.get_axis(2)
        if alpha==0 and self.first_throttle:
            alpha = 1
        else:
            self.first_throttle = False

        y = alpha*(-1/2.)+0.5
        return y

    def get_brake(self):
        ''' returns the brake [0,1] '''
        self.__sync()

        alpha = self.joystick.get_axis(3)
        if alpha==0 and self.first_brake:
            alpha = 1
        else:
            self.first_brake = False

        y = alpha*(-1/2.)+0.5
        return y

    def get_cartesian(self):
        ''' returns position of joystick in cartesian coordinates '''
        return (self.get_x(), self.get_y())

    def get_steering(self):
        return self.get_x()*180/3.1415*7.66

    def get_polar(self):
        ''' returns position of joystick in polar coordinates '''
        (x,y) = self.get_cartesian
        r = (x**2 + y**2)**0.5
        theta = np.arctan2(y, x)
        return (r,theta)

    def get_num_buttons(self):
        return self.joystick.get_numbuttons()

    def get_button_click(self, button=0):
        ''' returns if the button is currently clicked '''
        self.__sync()
        return self.joystick.get_button(button)




def test_joystick():
    import matplotlib.pyplot as plt

    plt.show()
    a = plt.gca()
    a.set_xlim(-1,1)
    a.set_ylim(-1,1)
    line, = a.plot(0,0,'-r', linewidth=3)

    joystick = Joystick()
    while True:
        (x,y) = joystick.get_cartesian()
        b = joystick.get_button_click()
        print x,y,b
        line.set_xdata([0,x])
        line.set_ydata([0,y])
        line.set_color('b' if b else 'r')
        plt.draw()
        plt.pause(1e-17)
    plt.show()

def test_steeringwheel():
    import cv2, urllib

    '''url = urllib.urlopen("https://d30y9cdsu7xlg0.cloudfront.net/png/22830-200.png")
    wheel_image = np.asarray(bytearray(url.read()), dtype="uint8")
    wheel_image = cv2.imdecode(wheel_image, cv2.IMREAD_COLOR)
    '''
    wheel_image = cv2.imread('../assets/img/mit.jpg',1)

    (n,m,d) = wheel_image.shape
    import pdb; pdb.set_trace()
    joystick = Joystick()
    while True:
        alpha = joystick.get_x()*180/np.pi*7.66
        print alpha
        M = cv2.getRotationMatrix2D((m/2,n/2),-alpha,1)
        dst = cv2.warpAffine(wheel_image,M,(m,n))
        cv2.imshow('wheel',cv2.resize(dst,None,fx=3,fy=3))
        cv2.waitKey(1)


def test_throttle_brake():
    import cv2, urllib

    joystick = Joystick()
    while True:
        throttle = joystick.get_throttle()
        brake = joystick.get_brake()
        print throttle, brake



# uncomment to run test case
#test_joystick()
# test_steeringwheel()
# test_throttle_brake()
