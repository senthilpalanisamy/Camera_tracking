import cv2
import numpy as np

def detect_mice_pose(image):
    print(image)
    print('here')
    print(image.shape)
    print('####')
    # cv2.imshow("image", image)
    # image = np.zeros((100, 100, 3))
    # image = cv2.imread('./build/test.png')
    cv2.imshow("image", image)
    cv2.waitKey(0)
    return None


if __name__=='__main__':
    detect_mice_pose(0)
