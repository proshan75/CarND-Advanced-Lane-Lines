import numpy as np
import cv2
import glob
import pickle

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# Make a list of calibration images
images = glob.glob('camera_cal/calibration*.jpg')

# Step through the list and search for chessboard corners
for index, filename in enumerate(images):
    image = cv2.imread(filename)
    print('processing filename: ', filename)

    if index == 1:
        h, w = image.shape[:2]
        print('image shape', image.shape[:2])

    #convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # find corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # draw the corners, save the image with corners identified
        cv2.drawChessboardCorners(image, (9,6), corners, ret)
        write_name= 'output_images/corners_found'+str(index+1)+'.jpg'
        print('corners found: ', write_name)
        cv2.imwrite(write_name, image)
    else:
        print('corners not found for image: ', filename)

# Do camera calibration given object points and image points
retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None )

#dst = cv2.undistort()
dist_pickle = {}
dist_pickle["cameraMatrix"] = cameraMatrix
dist_pickle["distCoeffs"] = distCoeffs
pickle.dump(dist_pickle, open('./calibration_pickle.p', 'wb'))


def undistort(filename,  cameraMatrix, distCoeffs):
    image = cv2.imread(filename)
    print('processing filename: ', filename)
    # undistort the image
    undistorted_image = cv2.undistort(image, cameraMatrix, distCoeffs, None, cameraMatrix)
    return undistorted_image


# Step through the list and search for chessboard corners
for index, filename in enumerate(images):
    image = undistort(filename, cameraMatrix, distCoeffs)
    write_name= 'output_images/chess_undistort/undistort_camera_cal'+str(index+1)+'.jpg'
    print('corners found: ', write_name)
    cv2.imwrite(write_name, image)


