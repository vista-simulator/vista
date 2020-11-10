import json
import numpy as np
import os
import requests
from simple_pid import PID
import subprocess as sp
from threading import Thread
import time

class LogitechDriver:
    def __init__(self, url="http://localhost", port=3001, max_angle=500, auto_button=False, delta_t=0.001, verbose=False):

        self.max_angle = float(max_angle)
        self.url = "{}:{}".format(url, port)
        cwd = os.path.dirname(os.path.realpath(__file__))
        cmd = ["sudo","node",os.path.join(cwd,"logitech-node.js"), str(port), str(max_angle), str(int(verbose))]
        self.pipe = sp.Popen(cmd, close_fds=True)
        self._header = {'Content-type': 'application/json', 'Accept': 'text/plain'}
        time.sleep(0.5)

        self.auto = False
        self._last_steering_angle = 0.
        self._done = False
        # self._pid = PID(Kp=0.03, Ki=0.00, Kd=0.001, output_limits=(-0.5, 0.5))
        self._pid = PID(Kp=0.2, Ki=0.0, Kd=0.004, setpoint=50, output_limits=(-0.5, 0.5), sample_time=delta_t)
        # self._pid = PID(Kp=0.2, Ki=0.01, Kd=0.008, setpoint=50, output_limits=(-0.5, 0.5), sample_time=delta_t)
        def _controller():
            while True:
                if self._done:
                    break
                if auto_button:
                    self.auto = self.get_auto_from_button()
                if not self.auto:
                    time.sleep(0.1)
                    continue

                control = self._pid(self._last_steering_angle)+0.5
                self._last_steering_angle = self.send_force(control)

                if not self.auto: # edge case when set auto comes right after the other check above (catch it here)
                    self.send_force(0.5) #no force

        self._thread = Thread(target=_controller)
        self._thread.start()

    def reset_auto_device(self):
        requests.post(self.url, data=json.dumps({'cmd':-1}), headers=self._header)

    def get_steering(self):
        r = requests.post(self.url, data=json.dumps({'cmd':0}), headers=self._header)
        return float(r.content)

    def get_throttle(self):
        r = requests.post(self.url, data=json.dumps({'cmd':1}), headers=self._header)
        return float(r.content)

    def get_auto_from_button(self):
        r = requests.post(self.url, data=json.dumps({'cmd':2}), headers=self._header)
        return bool(int(r.content))

    def send_force(self, force):
        r = requests.post(self.url, data=json.dumps({'cmd':3,'msg':force}), headers=self._header)
        return float(r.content)

    def set_steering(self, angle):
        self._pid.setpoint = angle * 50.0/self.max_angle + 50.
        return self._last_steering_angle

    def set_auto(self, auto):
        r = requests.post(self.url, data=json.dumps({'cmd':4,'msg':int(auto)}), headers=self._header)
        self.auto = auto

    def destroy(self):
        self._done = True
        time.sleep(0.1)
        self.pipe.kill()


def test_steeringwheel():
    import cv2, urllib

    '''url = urllib.urlopen("https://d30y9cdsu7xlg0.cloudfront.net/png/22830-200.png")
    wheel_image = np.asarray(bytearray(url.read()), dtype="uint8")
    wheel_image = cv2.imdecode(wheel_image, cv2.IMREAD_COLOR)
    '''
    wheel_image = cv2.imread('../assets/img/mit.jpg',1)
    wheel_image = cv2.resize(wheel_image, (200,200))
    (n,m,d) = wheel_image.shape
    wheel = LogitechDriver()
    while True:
        angle = wheel.get_steering()
        print angle
        M = cv2.getRotationMatrix2D((m/2,n/2),-angle,1)
        dst = cv2.warpAffine(wheel_image,M,(m,n))
        cv2.imshow('wheel',cv2.resize(dst,None,fx=3,fy=3))
        if cv2.waitKey(1) == 27: # break if ESC
           break
    wheel.destroy()


# dev = Logitech(auto_button=False)
# import pdb; pdb.set_trace()

# test_steeringwheel()
