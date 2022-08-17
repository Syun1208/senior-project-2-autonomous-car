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
        self.d = 0.05
        self.error_arr = np.zeros(5)
        self.arr_normal = []
        self.sign = sign
        self.center = 0
        self.bboxSize = 1000
        self.width = np.zeros(10)
        self.error = 0

    def checkLane(self):
        arr_normal = []
        height = 30
        lineRow = self.mask[height, :]
        for x, y in enumerate(lineRow):
            if y == 255:
                arr_normal.append(x)
        if not arr_normal:
            arr_normal = [91]
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
        self.time = time.time()
        D = (error - self.error_arr[1]) / delta_t * self.d
        I = np.sum(self.error_arr) * delta_t * self.i
        angle = P + I + D
        if abs(angle) > 3:
            angle = np.sign(angle) * 40
        return int(angle)

    def turnLeft(self):
        Min, Max = self.checkLane()
        if time.time() - self.time <= 1:
            self.speedDecrease()
            self.center = int(Min + Max) * 3 / 4
        return self.center

    def turnRight(self):
        if time.time() - self.time < 1:
            self.speedDecrease()
            self.center = 120
        return self.center

    def straight(self):
        Min, Max = self.checkLane()
        if time.time() - self.time < 1:
            if 100 <= Max <= 150 and 2 <= Min <= 70:
                self.width[1:] = self.width[0:-1]
                if Max - Min > 60:
                    self.width[0] = Max - Min
            widthRoad = np.average(self.width)
            if Max < 105 and 25 <= Min <= 55 or Max >= 120 and 25 <= Min <= 55:
                self.center = Min + int(widthRoad / 2)
            elif Min >= 55 and 105 <= Max <= 120 or Min < 25 and 105 <= Max <= 120:
                self.center = Max - int(widthRoad / 2)
            self.center = int(Min + Max) / 2
        return self.center

    def obstacleAvoiding(self):
        Min, Max = self.checkLane()
        self.speedDecrease()
        if self.sign == 'carright':
            self.center = Max - 25
        elif self.sign == 'carleft':
            self.center = Min + 20
        return self.center

    def speedDecrease(self):
        self.currentSpeed = 5

    def speedIncrease(self):
        self.currentSpeed = 70

    def trafficSignsController(self):
        distance = self.houghLine()
        Min, Max = self.checkLane()
        if self.sign != 'straight' or self.sign == 'nostraight':
            self.speedDecrease()
            if time.time() - self.time >= 5:
                if self.sign == 'turnleft' or self.sign == 'noright' or Min <= 10:
                    self.center = self.turnLeft()
                elif self.sign == 'turnright' or self.sign == 'noleft' or Max >= 150:
                    self.center = self.turnRight()
                else:
                    self.center = self.obstacleAvoiding()
        else:
            if time.time() - self.time >= 5:
                if self.sign == 'noright' or self.sign == 'noleft' or self.sign == 'unknown' or not self.sign:
                    self.speedIncrease()
                    self.center = self.straight()
        self.error = int(self.mask.shape[1] / 2) - self.center
        return self.error
