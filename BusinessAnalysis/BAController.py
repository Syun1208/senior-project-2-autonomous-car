from BusinessAnalysis.BAImageProcessing import imageProcessing
import numpy as np
import time

'''Objective: return error and sendBackSpeed'''


class Controller(imageProcessing):
    def __init__(self, mask, startTime, currentSpeed, sign, timeCar):
        super(Controller, self).__init__(mask)
        self.mask = mask
        self.time = startTime
        self.timeCar = timeCar
        self.currentSpeed = currentSpeed
        self.p = 0.32
        self.i = 0
        self.d = 0.1
        self.error_arr = np.zeros(5)
        self.arr_normal = []
        if sign is None:
            self.sign = 'empty'
            self.bboxSize = 0
        else:
            self.sign = sign[1]
            self.bboxSize = sign[2]
        self.center = 0
        self.width = np.zeros(10)
        self.error = 0

    def checkLane(self):
        height = 18
        lineRow = self.mask[height, :]
        for x, y in enumerate(lineRow):
            if y == 255:
                self.arr_normal.append(x)
        if not self.arr_normal:
            self.arr_normal = [50, 110]
        Min_Normal = min(self.arr_normal)
        Max_Normal = max(self.arr_normal)

        return Min_Normal, Max_Normal

    def computeError(self):
        height = 18
        lineRow = self.mask[height, :]
        for x, y in enumerate(lineRow):
            if y == 255:
                self.arr_normal.append(x)
        if not self.arr_normal:
            self.arr_normal = [50, 110]
        Min = min(self.arr_normal)
        Max = max(self.arr_normal)
        self.center = int((Min + Max) / 2)
        self.error = int(self.mask.shape[1] / 2) - self.center
        return self.error

    def PIDController(self, error):
        # global pre_Time
        self.error_arr[1:] = self.error_arr[0:-1]
        self.error_arr[0] = error
        P = error * self.p
        delta_t = time.time() - self.time
        print('PID Time: ', delta_t)
        self.time = time.time()
        D = (error - self.error_arr[1]) / delta_t * self.d
        I = np.sum(self.error_arr) * delta_t * self.i
        angle = P + I + D
        if abs(angle) > 5:
            angle = np.sign(angle) * 30
        return int(angle)

    def turnLeft(self):
        Min, Max = self.checkLane()
        self.speedDecrease()
        # if time.time() - self.timeCar >= 3 or self.bboxSize >= 3000 or self.houghLine() <= 65:
        self.center = int((Min + Max) * 1 / 8)
        return self.center

    def turnRight(self):
        Min, Max = self.checkLane()
        if time.time() - self.timeCar <= 0.1 or self.bboxSize >= 1200 or self.houghLine() <= 70:
            self.speedDecrease()
            self.center = int(Min + Max) * 3 / 4
        return self.center

    def straight(self):
        Min, Max = self.checkLane()
        # if 100 <= Max <= 150 and 2 <= Min <= 70:
        #     self.width[1:] = self.width[0:-1]
        #     if Max - Min > 60:
        #         self.width[0] = Max - Min
        # widthRoad = np.average(self.width)
        # if Max < 105 and 25 <= Min <= 55 or Max >= 120 and 25 <= Min <= 55:
        #     self.center = Min + int(widthRoad / 2)
        # elif Min >= 55 and 105 <= Max <= 120 or Min < 25 and 105 <= Max <= 120:
        #     self.center = Max - int(widthRoad / 2)
        self.center = int(Min + Max) / 2
        return self.center

    def obstacleAvoiding(self):
        Min, Max = self.checkLane()
        self.speedDecrease()
        if self.sign == 'carright':
            self.center = int(Min + Max) * 3 / 8
        elif self.sign == 'carleft':
            self.center = int(Min + Max) * 1 / 8
        return self.center

    def speedDecrease(self):
        self.currentSpeed = -5

    def speedIncrease(self):
        self.currentSpeed = 90

    def trafficSignsController(self):
        distance = self.houghLine()
        Min, Max = self.checkLane()
        if self.sign != 'straight' or self.sign == 'nostraight':
            self.speedDecrease()
            if self.sign == 'turnleft' or self.sign == 'noright':
                self.center = self.turnLeft()
            elif self.sign == 'turnright' or self.sign == 'noleft':
                self.center = self.turnRight()
            else:
                self.center = self.obstacleAvoiding()
        else:
            self.speedDecrease()
            if self.sign == 'noright' or self.sign == 'noleft' or self.sign == 'unknown' or not self.sign:
                if time.time() - self.timeCar >= 5:
                    self.speedIncrease()
                    self.center = self.straight()
        print('Center: ', self.center)
        self.error = int(self.mask.shape[1] / 2) - self.center
        return self.error
