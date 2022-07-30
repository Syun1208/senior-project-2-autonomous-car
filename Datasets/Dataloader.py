import socket
import cv2
import numpy as np


class Map:
    def __init__(self, currentAngle, currentSpeed, sendBackAngle, sendBackSpeed):
        self.PORT = 54321
        self.currentAngle = currentAngle
        self.currentSpeed = currentSpeed
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sendBackAngle = sendBackAngle
        self.sendBackSpeed = sendBackSpeed

    def socketClose(self):
        self.s.close()

    def Connect(self):
        self.s.connect(('127.0.0.1', self.PORT))
        message_getState = bytes("0", "utf-8")
        self.s.sendall(message_getState)
        state_date = self.s.recv(100)

        self.currentSpeed, self.currentAngle = state_date.decode(
            "utf-8"
        ).split(' ')

        message = bytes(f"1 {self.sendBackAngle} {self.sendBackSpeed}", "utf-8")
        self.s.sendall(message)
        data = self.s.recv(100000)
        image = cv2.imdecode(
            np.frombuffer(
                data,
                np.uint8
            ), -1
        )
        return image, self.currentAngle, self.currentSpeed, self.sendBackSpeed, self.sendBackAngle
