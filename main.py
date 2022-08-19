import time
import numpy as np
import cv2
import torch
from model.UNET import build_unet
from model.CNN import Network
from sklearn.ensemble import RandomForestRegressor
from weights.loadWeights import weights
from BusinessAnalysis.BASegmentations import segmentation
from BusinessAnalysis.BADetections import detection
from BusinessAnalysis.BAController import Controller
from BusinessAnalysis.BAImageProcessing import imageProcessing
from Datasets.Dataloader import Map

# from sklearn.ensemble import RandomForestRegressor

global M


def main():
    global M
    currentAngle = 0
    currentSpeed = 0
    sendBackSpeed = 0
    sendBackAngle = 0
    MAX_SPEED = 60
    SAVE_SPEED = 50
    delaySign = None
    signArr = np.zeros(5)
    delayTime = 3
    count = 0
    t0 = 0
    pretrainedModel = weights()
    predictedUNET = pretrainedModel.modelUNET()
    predictedYOLOv5m = pretrainedModel.modelYOLOv5m()
    predictedCNN = pretrainedModel.modelCNN()
    start = time.time()
    try:
        while True:
            M = Map(currentAngle, currentSpeed, sendBackAngle, sendBackSpeed)
            image, currentAngle, currentSpeed, sendBackAngle, sendBackSpeed = M.Connect()
            '''-------------------------Work Space----------------------------'''
            modelUNET = segmentation(image)
            pretrainedUNET = modelUNET.predict(predictedUNET)
            IP = imageProcessing(pretrainedUNET)
            pretrainedUNET = IP.removeSmallContours()
            cv2.imshow('Origin mask', pretrainedUNET)
            modelYOLOv5m = detection(image)
            sign = modelYOLOv5m.predict(predictedYOLOv5m, predictedCNN)
            '''-------------------------Controller----------------------------'''
            sendBackSpeed = 40
            preTime = time.time()
            balance = Controller(pretrainedUNET, start, sendBackSpeed, sign, preTime)
            # Min, Max = balance.checkLane()
            # print('Min: ', Min)
            # print('Max: ', Max)
            # if sign:
            #     print('CNN: ', sign[0])
            #     error = balance.trafficSignsController()
            #     if sign[1] != 'straight' or sign[1] != 'carleft' or sign[1] != 'carright' or sign[1] != 'unknown':
            #         while timer:
            #             delayTime += 1
            #             print('Delay time: ', delayTime)
            #             if delayTime == 2300 and sign[2] > 4600:
            #                 print('Rẽ đi má')
            #                 sendBackAngle = - balance.PIDController(error) * 16 / 19
            #                 timer = False
            #             elif not sign:
            #                 delayTime += 1
            #         if Min <= 5 or Max >= 150:
            #             sendBackSpeed = -1
            #             error = balance.computeError()
            #             sendBackAngle = - balance.PIDController(error) * 8 / 19
            #         else:
            #             sendBackSpeed = MAX_SPEED
            #     else:
            #         sendBackSpeed = -1
            #         sendBackAngle = - balance.PIDController(error) * 12 / 19
            # else:
            #     error = balance.computeError()
            #     if Min <= 1 or Max >= 155:
            #         sendBackSpeed = -1
            #         sendBackAngle = - balance.PIDController(error) * 10 / 19
            #     else:
            #         sendBackAngle = - balance.PIDController(error) * 5 / 19
            #     if -4 >= sendBackAngle or sendBackAngle >= 4:
            #         sendBackSpeed = -1
            #     elif -1 <= sendBackAngle <= 1:
            #         sendBackSpeed = MAX_SPEED
            #     if sendBackSpeed < 1 and time.time() - start < 2:
            #         sendBackSpeed = MAX_SPEED
            # if Min <= 1 and Max >= 155:
            #     sendBackSpeed = -3
            #     error = balance.computeError()
            #     sendBackAngle = - balance.PIDController(error) * 16 / 19
            if sign:
                # if sign != 'carright' or sign != 'carleft':
                #     pretrainedUNET = balance.trafficSignsControllerByCropImage()
                #     balance = Controller(pretrainedUNET, start, sendBackSpeed, sign, preTime)
                #     error = balance.computeError()
                #     sendBackAngle = - balance.PIDController(error) * 35 / 60
                # else:
                #     center = balance.obstacleAvoiding()
                #     error = int(pretrainedUNET.shape[1] / 2) - center
                #     sendBackAngle = - balance.PIDController(error) * 18 / 60
                sendBackSpeed = -1
                error = balance.trafficSignsControllerByCropImage()
                sendBackAngle = - balance.PIDController(error) * 35 / 60
                t0 = time.time()
                delaySign = sign[1]
            else:
                if delaySign is str or delaySign is not None:
                    sendBackSpeed = -1
                    balance = Controller(pretrainedUNET, start, sendBackSpeed, delaySign, preTime)
                    error = balance.trafficSignsControllerByCropImage()
                    sendBackAngle = - balance.PIDController(error) * 35 / 60
                    # if delaySign != 'carright' or delaySign != 'carleft':
                    #     sendBackSpeed = -1
                    #     center = balance.obstacleAvoiding()
                    #     error = int(pretrainedUNET.shape[1] / 2) - center
                    #     sendBackAngle = - balance.PIDController(error) * 18 / 60
                    # else:
                    #     sendBackSpeed = -2
                    #     print('Dang delay')
                    #     balance = Controller(pretrainedUNET, start, sendBackSpeed, delaySign, preTime)
                    #     pretrainedUNET = balance.trafficSignsControllerByCropImage()
                    #     balance = Controller(pretrainedUNET, start, sendBackSpeed, delaySign, preTime)
                    #     print('Dang delay 1')
                    #     error = balance.computeError()
                    #     if delaySign != 'straight':
                    #         sendBackAngle = - balance.PIDController(error) * 35 / 60
                    #     else:
                    #         sendBackAngle = - balance.PIDController(error) * 15 / 60
                else:
                    error = balance.computeError()
                    sendBackAngle = - balance.PIDController(error) * 16 / 60
                if time.time() - t0 >= delayTime:
                    delaySign = None
                    t0 = 0
            if -4 >= sendBackAngle or sendBackAngle >= 4:
                sendBackSpeed = -1
            elif -1 <= sendBackAngle <= 1:
                sendBackSpeed = MAX_SPEED
            if sendBackSpeed < 1 and time.time() - start < 2:
                sendBackSpeed = MAX_SPEED
            cv2.imshow('Origin mask', pretrainedUNET)
            print('CNN: ', sign)
            print('Delay Sign: ', delaySign)
            print('=====================')
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        M.socketClose()


if __name__ == '__main__':
    main()
