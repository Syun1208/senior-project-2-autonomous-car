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
    MAX_SPEED = 70
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
            cv2.imshow('predict', pretrainedUNET)
            IP = imageProcessing(pretrainedUNET)
            pretrainedUNET = IP.removeSmallContours()
            ROI = IP.ROI()
            cv2.imshow('ROI', ROI)
            print('Distances: ', IP.houghLine())
            modelYOLOv5m = detection(image)
            # sign, signYOLO, bboxSize = modelYOLOv5m.predict(predictedYOLOv5m, predictedCNN)
            sign = modelYOLOv5m.predict(predictedYOLOv5m, predictedCNN)
            '''-------------------------Controller----------------------------'''
            # balance = Controller(pretrainedUNET, start, sendBackSpeed, sign, bboxSize)
            sendBackSpeed = 30
            # if sign is None or len(sign) != 0 or not sign:
            #     sign = list(['empty', 'empty', 0])
            print('CNN: ', sign)
            balance = Controller(pretrainedUNET, start, sendBackSpeed, sign)
            if sign:
                sendBackSpeed = 2
                error = balance.trafficSignsController()
                sendBackAngle = - balance.PIDController(error) * 12 / 19
                sendBackSpeed = 10
            else:
                error = balance.computeError()
                sendBackAngle = - balance.PIDController(error) * 6 / 19
                # x = np.array([-12, -1, -0.25, 0,
                #               0.25, 1, 12])
                # y = np.array([0, 10, 25, 55,
                #               25, 10, 0])
                #
                # x = x.reshape(-1, 1)
                #
                # regressor = RandomForestRegressor(n_estimators=300, random_state=0)
                # regressor.fit(np.array(x), np.array(y))
                #
                # sendBackSpeed = np.mean(regressor.predict([[sendBackAngle]]))
                if abs(sendBackAngle) >= 3 and sendBackSpeed >= 10 or abs(error) >= 10:
                    sendBackSpeed = 10
                elif abs(sendBackAngle) >= 1:
                    sendBackSpeed = MAX_SPEED
                if sendBackSpeed <= 15:
                    sendBackSpeed = MAX_SPEED
                # else:
                #     sendBackSpeed = sendBackSpeed - 20
            print(error)
            print(sendBackAngle)
            print(sendBackSpeed)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        M.socketClose()


if __name__ == '__main__':
    main()
