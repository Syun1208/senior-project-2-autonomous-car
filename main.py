import time
import numpy as np
import cv2
import torch
from model.UNET import build_unet
from model.CNN import Network
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
    MAX_SPEED = 45.0
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
            modelYOLOv5m = detection(image)
            # sign, signYOLO, bboxSize = modelYOLOv5m.predict(predictedYOLOv5m, predictedCNN)
            sign = modelYOLOv5m.predict(predictedYOLOv5m, predictedCNN)
            '''-------------------------Controller----------------------------'''
            # balance = Controller(pretrainedUNET, start, sendBackSpeed, sign, bboxSize)
            sendBackSpeed = 25
            if sign is None:
                sign = ['unknown', 'unknown', 0]
            print('CNN: ', sign)
            balance = Controller(pretrainedUNET, start, sendBackSpeed, sign)
            if sign:
                time.sleep(5)
                error = balance.trafficSignsController()
                sendBackAngle = - balance.PIDController(error) * 8 / 19
            else:
                error = balance.computeError()
                sendBackAngle = - balance.PIDController(error) * 4 / 19
                if -20 >= error >= 15:
                    sendBackSpeed = 10
                elif -4 <= error <= 1:
                    sendBackSpeed = 60
                # sendBackAngle = - balance.PIDController(error) * 5 / 19
                print(error)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        M.socketClose()


if __name__ == '__main__':
    main()
