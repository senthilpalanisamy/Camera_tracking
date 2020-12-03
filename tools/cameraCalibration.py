import numpy as np
import cv2 as cv
import glob
import json

import os
import sys
# termination criteria

output_base_path = './config/camera_intrinsics_1024x1024/'
if(not os.path.isdir(output_base_path)):
    sys.exit('please provide a correct path for writing results')

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
camera_no=0
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
base_input_path='/home/senthil/work/Camera_tracking/samples/11_30/lens_dist_inside/'+str(camera_no)+'/'
images1 = glob.glob(base_input_path+'*.png')
images2 = glob.glob(base_input_path+'*.jpg')
images2 = glob.glob(base_input_path+'*.bmp')
images = sorted(images1 + images2)
print(images[0])

for index, fname in enumerate(images):
    print(index)
    gray = cv.imread(fname, 0)
    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (9,6), None)
    # If found, add object points, image points (after refining them)
    # print('here')
    if ret == True:
        # print('there')
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv.drawChessboardCorners(gray, (9,6), corners2, ret)
        #cv.imshow('img', gray)
        #cv.waitKey(5)

cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
R = cv.Rodrigues(rvecs[0])[0]
Homography = np.zeros((3, 3))
TransformMatrix = np.zeros((4,4))
TransformMatrix[:3, :3] = R
TransformMatrix[3, 3] = 1
TransformMatrix[:3, 3] = tvecs[0].reshape(3)
projection_matrix = np.zeros((3,4))
projection_matrix[:3, :3] = np.identity(3)
CameraMatrix = mtx.dot(projection_matrix).dot(TransformMatrix)
Homography[:, :2] = CameraMatrix[:, :2]
Homography[:, 2] = CameraMatrix[:, 3]

print(mtx)
print(dist)


data = {}
data['intrinsic'] = mtx.tolist()
data['dist'] = dist.tolist()
# data['homography'] = Homography.tolist()
# data['extrinsic'] = TransformMatrix.tolist()
output_path = output_base_path + 'camera_'+ str(camera_no) + '.json'

with open(output_path, 'w') as f:
    json.dump(data, f)

