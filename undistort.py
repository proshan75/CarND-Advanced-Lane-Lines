import pickle
import cv2
import numpy as np
import glob

# Step through the test images list and undistort

def undistort(filename,  cameraMatrix, distCoeffs):
    image = cv2.imread(filename)
    print('processing filename: ', filename)
    # undistort the image
    undistorted_image = cv2.undistort(image, cameraMatrix, distCoeffs, None, cameraMatrix)
    return undistorted_image
