from weights.loadWeights import weights
from BusinessAnalysis.BARecognitions import recognition
import torch
import cv2


class detection:
    def __init__(self, image):
        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def predict(self, modelDetection, modelRecognition):
        # self.image = self.image[125:, :]
        self.image = cv2.resize(self.image, (640, 360))
        # pretrainedYOLOv5m = weights(self.image)
        resultsDetection = modelDetection(self.image)
        coordinateDetection = resultsDetection.pandas().xyxy[0]
        '''Calculate predicted bounding box'''
        if len(coordinateDetection) != 0:
            if float(coordinateDetection.confidence[0]) >= 0.9:
                x_min = int(resultsDetection.xyxy[0][0][0])
                y_min = int(resultsDetection.xyxy[0][0][1])
                x_max = int(resultsDetection.xyxy[0][0][2])
                y_max = int(resultsDetection.xyxy[0][0][3])
                imgBoundingBox = self.image[y_min:y_max, x_min:x_max]
                resultsRecognition = recognition(imgBoundingBox)
                coordinateRecognition = resultsRecognition.predict(modelRecognition)
                bboxSize = (x_max - x_min) * (y_max - y_min)
                if coordinateDetection.name[0] == 'car':
                    if x_max >= 200 and y_max >= 200:
                        coordinateDetection.name[0] = 'carleft'
                        coordinateRecognition = 'carleft'
                    coordinateDetection.name[0] = 'carright'
                    coordinateRecognition = 'carright'
                elif coordinateRecognition == 'unknown':
                    coordinateDetection.name[0] = 'unknown'
                if coordinateRecognition is None and coordinateDetection.name[0] is None:
                    coordinateRecognition = 'empty'
                    coordinateDetection.name[0] = 'empty'
                return list([coordinateRecognition, coordinateDetection.name[0], bboxSize])
        # else:
        #     return None
