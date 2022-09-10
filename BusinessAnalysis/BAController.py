from BusinessAnalysis.BAImageProcessing import imageProcessing
import numpy as np
import time

'''Objective: return error and sendBackSpeed'''


class Controller(imageProcessing):
    def __init__(self, mask, startTime, currentSpeed, sign):
        super(Controller, self).__init__(mask)
        self.mask = mask
        self.time = startTime
        self.currentSpeed = currentSpeed
        self.p = 0.35
        self.i = 0
        self.d = 0.01
        self.error_arr = np.zeros(5)
        self.arr_normal = []
        if sign is None:
            self.sign = 'empty'
            self.bboxSize = 0
        else:
            self.sign = sign[1]
            self.bboxSize = sign[2]
        self.width = np.zeros(10)

    def checkLane(self):
        arr_normal = []
        height = 20
        lineRow = self.mask[height, :]
        for x, y in enumerate(lineRow):
            if y == 255:
                arr_normal.append(x)
        if not arr_normal:
            arr_normal = [40, 120]
        Min_Normal = min(arr_normal)
        Max_Normal = max(arr_normal)

        return Min_Normal, Max_Normal

    def computeError(self):
        height = 18
        lineRow = self.mask[height, :]
        for x, y in enumerate(lineRow):
            if y == 255:
                self.arr_normal.append(x)
        if not self.arr_normal:
            self.arr_normal = [40, 120]
        Min = min(self.arr_normal)
        Max = max(self.arr_normal)
        center = int((Min + Max) / 2)
        error = int(self.mask.shape[1] / 2) - center
        return error

    def PIDController(self, error):
        self.error_arr[1:] = self.error_arr[0:-1]
        self.error_arr[0] = error
        P = error * self.p
        delta_t = time.time() - self.time
        D = (error - self.error_arr[1]) / delta_t * self.d
        I = np.sum(self.error_arr) * delta_t * self.i
        angle = P + I + D
        if abs(angle) > 5:
            angle = np.sign(angle) * 40
        return int(angle)

    def turnLeft(self):
        center = 0
        if time.time() - self.time < 1:
            self.speedDecrease()
            center = 5
        return center

    def turnRight(self):
        center = 0
        if time.time() - self.time < 1:
            self.speedDecrease()
            center = 120
        return center

    def straight(self):
        Min, Max = self.checkLane()
        error = 0
        widthRoad = 0
        center = 0
        if time.time() - self.time < 2:
            if 100 <= Max <= 150 and 2 <= Min <= 70:
                self.width[1:] = self.width[0:-1]
                if Max - Min > 60:
                    self.width[0] = Max - Min
            widthRoad = np.average(self.width)
            if Max < 100 and 30 <= Min <= 60 or Max >= 120 and 30 <= Min <= 60:
                center = Min + int(widthRoad / 2)
            elif Min >= 60 and 100 <= Max <= 120 or Min < 30 and 100 <= Max <= 120:
                center = Max - int(widthRoad / 2)
            center = int(Min + Max) / 2
        return center

    def obstacleAvoiding(self):
        Min, Max = self.checkLane()
        center = 0
        self.speedDecrease()
        if self.sign == 'carright':
            center = Max - 25
        elif self.sign == 'carleft':
            center = Min + 20
        return center

    def speedDecrease(self):
        self.currentSpeed = 1

    def speedIncrease(self):
        self.currentSpeed = 70

    def trafficSignsController(self):
        distance = self.houghLine()
        Min, Max = self.checkLane()
        center = 0
        if self.sign != 'straight' or self.sign == 'nostraight':
            self.speedDecrease()
            if self.sign == 'turnleft' or self.sign == 'noright' or Min <= 10:
                center = self.turnLeft()
            elif self.sign == 'turnright' or self.sign == 'noleft' or Max >= 150:
                center = self.turnRight()
            else:
                center = self.obstacleAvoiding()
        else:
            if self.sign == 'noright' or self.sign == 'noleft' or self.sign != 'unknown' or self.sign:
                self.speedIncrease()
                center = self.straight()
        error = int(self.mask.shape[1] / 2) - center
        return error
