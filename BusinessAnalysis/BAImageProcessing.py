import cv2
import numpy as np
import time
import matplotlib.pyplot as plt


class imageProcessing:
    def __init__(self, mask):
        self.mask = mask

    def gaussianBlur(self):
        blur = cv2.GaussianBlur(self.mask, (5, 5), 0)
        return blur

    def canny(self):
        return cv2.Canny(self.gaussianBlur(), 50, 100)

    def ROI(self):
        height = self.canny().shape[0]
        width = self.canny().shape[1]
        polygon = np.array([
            [(0, height), (20, 50), (0, 50), (0, 20), (width, 20), (width, 50), (130, 50), (width, height)]
        ])
        imgMask = np.zeros_like(self.canny())
        cv2.fillPoly(imgMask, polygon, 255)
        cropped_image = cv2.bitwise_and(self.canny(), imgMask)
        return cropped_image

    def ROITurnRight(self):
        height = self.canny().shape[0]
        width = self.canny().shape[1]
        polygon = np.array([
            [(60, height), (width, height), (width, 0), (90, 12)]
        ])
        polygonLine = np.array([
            [(60, height), (90, 12)]
        ])
        polygonConcat = np.array([
            [(60, height), (width, height), (width, 0), (90, 12)]
        ])
        imgMaskLine = np.zeros_like(self.canny())
        cv2.fillPoly(imgMaskLine, polygonLine, 255)
        cropped_image_line = cv2.bitwise_or(self.canny(), imgMaskLine)

        imgMask = np.zeros_like(self.canny())
        cv2.fillPoly(imgMask, polygon, 255)
        cropped_image = cv2.bitwise_and(self.canny(), imgMask)

        result = cv2.bitwise_or(cropped_image, cropped_image_line)

        imgMaskConcat = np.zeros_like(self.canny())
        cv2.fillPoly(imgMaskConcat, polygonConcat, 255)
        cropped_image_concat = cv2.bitwise_and(result, imgMaskConcat)
        return cropped_image_concat

    def ROITurnLeft(self):
        height = self.canny().shape[0]
        width = self.canny().shape[1]
        polygon = np.array([
            [(80, height), (0, height), (0, 0), (90, 0)]
        ])
        polygonLine = np.array([
            [(60, height), (80, 15)]
        ])
        polygonConcat = np.array([
            [(90, height), (0, height), (0, 12), (80, 0)]
        ])
        imgMask = np.zeros_like(self.canny())
        cv2.fillPoly(imgMask, polygon, 255)
        cropped_image = cv2.bitwise_and(imgMask, self.canny())

        imgMaskLine = np.zeros_like(self.canny())
        cv2.fillPoly(imgMaskLine, polygonLine, 255)
        cropped_image_line = cv2.bitwise_or(self.canny(), imgMaskLine)

        result = cv2.bitwise_or(cropped_image, cropped_image_line)

        imgMaskConcat = np.zeros_like(self.canny())
        cv2.fillPoly(imgMaskConcat, polygonConcat, 255)
        cropped_image_concat = cv2.bitwise_and(result, imgMaskConcat)
        return cropped_image_concat

    def ROIStraight(self):
        height = self.canny().shape[0]
        width = self.canny().shape[1]
        polygon = np.array([
            [(0, 60), (60, 20), (100, 20), (width, height)]
        ])
        polygonLineLeft = np.array([
            [(20, 50), (60, 0)]
        ])
        polygonLineRight = np.array([
            [(140, 65), (100, 0)]
        ])
        polygonConcat = np.array([
            [(0, 60), (60, 20), (100, 20), (width, height)]
        ])
        imgMask = np.zeros_like(self.canny())
        cv2.fillPoly(imgMask, polygon, 255)
        cropped_image = cv2.bitwise_and(self.canny(), imgMask)

        imgMaskLineRight = np.zeros_like(self.canny())
        cv2.fillPoly(imgMaskLineRight, polygonLineRight, 255)
        cropped_image_right = cv2.bitwise_or(self.canny(), imgMaskLineRight)

        imgMaskLineLeft = np.zeros_like(self.canny())
        cv2.fillPoly(imgMaskLineLeft, polygonLineLeft, 255)
        cropped_image_left = cv2.bitwise_or(self.canny(), imgMaskLineLeft)

        result = cv2.bitwise_or(cropped_image_left, cropped_image_right)
        finalResult = cv2.bitwise_or(cropped_image, result)

        imgMaskConcat = np.zeros_like(self.canny())
        cv2.fillPoly(imgMaskConcat, polygonConcat, 255)
        cropped_image_concat = cv2.bitwise_and(finalResult, imgMaskConcat)

        return cropped_image_concat

    def removeSmallContours(self):
        image_binary = np.zeros((self.mask.shape[0], self.mask.shape[1]), np.uint8)
        contours = cv2.findContours(self.mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
        masked = cv2.drawContours(image_binary, [max(contours, key=cv2.contourArea)], -1, (255, 255, 255), -1)
        image_remove = cv2.bitwise_and(self.mask, self.mask, mask=masked)
        return image_remove

    def houghLine(self):
        lines = cv2.HoughLinesP(self.canny(), 1, np.pi / 4, 1, np.array([]), minLineLength=10, maxLineGap=10)
        lines1 = []
        o_d = 65
        lines2 = []
        o_intercept = 0
        h = 0
        if lines is not None:
            for line in lines:
                if line is None:
                    continue
                for x1, y1, x2, y2 in line:
                    if x1 == x2:
                        continue

                    slope = (y2 - y1) / (x2 - x1)
                    intercept = y1 - slope * x1

                    if slope == 0 and intercept > 5:
                        h += 1
                        if h == 1:
                            o_intercept = intercept

                        if intercept - o_intercept <= 5:
                            lines1.append([intercept])

                        else:
                            continue
                        lines2.append([intercept])
                        o_intercept = intercept
        if len(lines1) != 0:
            avg1 = np.sum(lines1) / len(lines1)
            d = 80 - avg1
            return d
        #     if d - o_d < 3:
        #         o_d = d
        #         return d
        #     else:
        #         return 200
        # else:
        #     o_d = 65
        #     return 200

    def houghLineTurnRight(self):
        pass

    def houghLineTurnLeft(self):
        pass

    def houghLineStraight(self):
        pass


if __name__ == '__main__':
    img = cv2.imread('D:\\SeniorProject2\\242_iuMinhAnh30000.png')
    mask = np.copy(img)
    binaryImg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ImageProcessing = imageProcessing(mask=img)
    # # img_resize = cv2.resize(ImageProcessing.ROI(), (1000, 500))
    cv2.imshow('ROI', ImageProcessing.ROI())
    plt.imshow(ImageProcessing.canny())
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
