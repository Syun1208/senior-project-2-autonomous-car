from weights.loadWeights import weights
import torch
import cv2


class recognition:
    def __init__(self, image):
        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.classes = ['noleft', 'noright', 'nostraight', 'straight', 'turnleft', 'turnright', 'unknown']

    def predict(self, model):
        img_rgb = cv2.resize(self.image, (64, 64))
        img_rgb = img_rgb / 255
        img_rgb = img_rgb.astype('float32')
        img_rgb = img_rgb.transpose(2, 0, 1)

        img_rgb = torch.from_numpy(img_rgb).unsqueeze(0)

        with torch.no_grad():
            img_rgb = img_rgb.to(self.device)
            # pretrainedCNN = weights(img_rgb)
            predictedImage = model(img_rgb)
            _, predicted = torch.max(predictedImage, 1)
            predicted = predicted.data.cpu().numpy()
            # return name of classes
            classPredict = self.classes[predicted[0]]
        return classPredict
