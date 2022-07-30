from model.UNET import build_unet
from model.CNN import Network
import torch
import cv2

import torch
import torch.nn as nn


class weights:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def modelUNET(self):
        predictedUNET = build_unet()
        predictedUNET = predictedUNET.to(self.device)
        predictedUNET.load_state_dict(torch.load('D:\\SeniorProject2\\weights\\my_checkpoints.pth',
                                                 map_location=self.device))
        predictedUNET.eval()
        return predictedUNET

    def modelYOLOv5m(self):
        predictedYOLOv5m = torch.hub.load('ultralytics/yolov5', 'custom', path='D:\\SeniorProject2\\weights\\best.pt')
        predictedYOLOv5m.to(self.device)
        predictedYOLOv5m.eval()
        return predictedYOLOv5m

    def modelCNN(self):
        modelCNN = Network()
        modelCNN.to(self.device)
        modelCNN.load_state_dict(torch.load('D:\\SeniorProject2\\weights\\CNN.pth', map_location=self.device))
        modelCNN.eval()
        return modelCNN
