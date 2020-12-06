import numpy as np
import cv2
from cv2 import aruco

workdir = "./workdir/"
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
board = aruco.CharucoBoard_create(15, 15, 1, .8, aruco_dict)
imboard = board.draw((21260, 21260))
cv2.imwrite("chessboard_21260.png", imboard)
