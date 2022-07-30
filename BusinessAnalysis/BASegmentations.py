from weights.loadWeights import weights
import torch
import numpy as np
import cv2


class segmentation:
    def __init__(self, image):
        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def predict(self, model):
        self.image = self.image[125:, :]
        self.image = cv2.resize(self.image, (160, 80))
        x = torch.from_numpy(self.image)
        x = x.to(self.device)
        x = x.transpose(1, 2).transpose(0, 1)
        x = x / 255.0
        x = x.unsqueeze(0).float()
        with torch.no_grad():
            # pretrainedUNET = weights(x)
            predictImage = model(x)
            predictImage = torch.sigmoid(predictImage)
            predictImage = predictImage[0]
            predictImage = predictImage.squeeze()
            predictImage = predictImage > 0.5
            predictImage = predictImage.cpu().numpy()
            predictImage = np.array(predictImage, dtype=np.uint8)
            predictImage = predictImage * 255
        return predictImage
