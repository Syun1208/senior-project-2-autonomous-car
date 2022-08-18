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
    MAX_SPEED = 50
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
            print('Distances: ', IP.houghLine())
            modelYOLOv5m = detection(image)
            sign = modelYOLOv5m.predict(predictedYOLOv5m, predictedCNN)
            '''-------------------------Controller----------------------------'''
            sendBackSpeed = 50
            print('CNN: ', sign)
            preTime = time.time()
            timer = True
            delayTime = 0
            balance = Controller(pretrainedUNET, start, sendBackSpeed, sign, preTime)
            if sign:
                sendBackSpeed = -5
                error = balance.trafficSignsController()
                while timer:
                    delayTime += 1
                    print('Delay time: ', delayTime)
                    if delayTime == 2100:
                        print('Rẽ đi má')
                        sendBackAngle = - balance.PIDController(error) * 16 / 19
                        sendBackSpeed = 10
                        timer = False
                    elif not sign:
                        delayTime += 1
            else:
                error = balance.computeError()
                sendBackAngle = - balance.PIDController(error) * 5 / 15
                if -3 >= sendBackAngle or sendBackAngle >= 3:
                    sendBackSpeed = -0.1
                elif -1 <= sendBackAngle <= 1:
                    sendBackSpeed = MAX_SPEED
                if sendBackSpeed < 1 and time.time() - start < 1:
                    sendBackSpeed = MAX_SPEED
            print('Error: ', error)
            print('Angle: ', sendBackAngle)
            print('Speed', sendBackSpeed)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        M.socketClose()


if __name__ == '__main__':
    main()
