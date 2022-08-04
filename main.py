import time
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

global M
start = time.time()


def main():
    global M
    global start
    currentAngle = 0
    currentSpeed = 0
    sendBackSpeed = 0
    sendBackAngle = 0
    MAX_SPEED = 40
    start = time.time()
    pretrainedModel = weights()
    predictedUNET = pretrainedModel.modelUNET()
    predictedYOLOv5m = pretrainedModel.modelYOLOv5m()
    predictedCNN = pretrainedModel.modelCNN()
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
            sign = modelYOLOv5m.predict(predictedYOLOv5m, predictedCNN)
            print('CNN: ', sign)
            '''-------------------------Controller----------------------------'''
            balance = Controller(pretrainedUNET, start, currentSpeed, sign, bboxSize=1000)
            if sign:
                error = balance.trafficSignsController()
            else:
                error = balance.computeError()
            print(error)
            sendBackAngle = - balance.PIDController(error) * 3 / 13
            '''----------------------SOMETHING IS WRONG-----------------------'''
            # if sendBackAngle <= -3 or sendBackAngle >= 3:
            #     sendBackSpeed = balance.speedDecrease()
            #     sendBackAngle = sendBackAngle * 2
            '''---------------------------------------------------------------'''
            sendBackSpeed = 20
            print(sendBackAngle)
            print(sendBackSpeed)
            end = time.time()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        M.socketClose()


if __name__ == '__main__':
    main()
