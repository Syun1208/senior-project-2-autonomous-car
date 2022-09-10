import socket
import cv2
import numpy as np

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
PORT = 54321
s.connect(('127.0.0.1', PORT))


class Map:
    def __init__(self, currentAngle, currentSpeed, sendBackAngle, sendBackSpeed):
        self.PORT = 54321
        self.currentAngle = currentAngle
        self.currentSpeed = currentSpeed
        self.s = s
        self.sendBackAngle = sendBackAngle
        self.sendBackSpeed = sendBackSpeed

    def socketClose(self):
        self.s.close()

    def Connect(self):
        message_getState = bytes("0", "utf-8")
        self.s.sendall(message_getState)
        state_date = self.s.recv(1000)

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
#     # Import socket module
# def Map():
#     s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     PORT = 54321
#     s.connect(('127.0.0.1', PORT))
#     while True:
#         message_getState = bytes("0", "utf-8")
#         s.sendall(message_getState)
#         state_date = s.recv(100)
#
#         try:
#             current_speed, current_angle = state_date.decode(
#                         "utf-8"
#                     ).split(' ')
#             except Exception as er:
#                     print(er)
#                     pass
#
#             message = bytes(f"1 {sendBack_angle} {sendBack_Speed}", "utf-8")
#             s.sendall(message)
#             data = s.recv(100000)
#
#             try:
#                 image = cv2.imdecode(
#                 np.frombuffer(
#                             data,
#                             np.uint8
#                         ), -1
#                     )
