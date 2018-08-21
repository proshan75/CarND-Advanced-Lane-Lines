import glob
import pickle

import matplotlib.pyplot as plt
import numpy as np

import cv2
from measure_curve import (measure_curvature_real, xm_per_pix)
from sliding_window import (bottom_half_histogram, find_lane_pixels,
                            fit_polynomial)
from Threshold import (abs_sobel_thresh, color_threshold, dir_threshold,
                       mag_thresh)
from undistort import undistort

# Read in the saved objpoints and imgpoints
calibration_pickle = pickle.load( open( "calibration_pickle.p", "rb" ) )
cameraMatrix = calibration_pickle["cameraMatrix"]
distCoeffs = calibration_pickle["distCoeffs"]

# Collect images for undistortion
#test_images = glob.glob('test_images/straight_lines*.jpg')
test_images = glob.glob('test_images/test*.jpg')


for index, filename in enumerate(test_images):
    print('processing image: ', filename)
    image = undistort(filename, cameraMatrix, distCoeffs)

    # Choose a Sobel kernel size
    ksize = 13 # Choose a larger odd number to smooth gradient measurements

    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh_min=20, thresh_max=200)
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh_min=50, thresh_max=200)
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(60, 130))
    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.7, 1.3))
    c_binary = color_threshold(image, sthreshold=(90,255), vthreshold=(90,255))

    combined = np.zeros_like(image[:,:,0])
    #combined[((gradx == 1) & (grady == 1) ) | ((mag_binary == 1) & (dir_binary == 1)) ] = 255 #
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (c_binary == 1) ] = 255 #
    #combined[((gradx == 1) & (grady == 1)) ] = 1
    #combined[(dir_binary == 1)] = 255
    #combined[(c_binary == 1)] = 255

    h, w = image.shape[:2]
    trap_top_width_per = 0.032
    trap_bot_width_per = 0.35
    trap_top_height_per = 0.62
    trap_bot_height_per = 0.98
    top_left = [w* (0.5 - trap_top_width_per) , h * trap_top_height_per]
    top_right = [w* (0.5 + trap_top_width_per), h * trap_top_height_per]
    bottom_left = [w* (0.5 - trap_bot_width_per), h * trap_bot_height_per]
    bottom_right = [w* (0.5 + trap_bot_width_per), h * trap_bot_height_per]
    offset = w * 0.25

    #result = combined_lines
    result = combined
    write_name= 'output_images/threshold/multi_thresholded_'+str(index+1)+'.jpg'
    print('threshold image: ', write_name)
    cv2.imwrite(write_name, result)

    src = np.float32([top_left, top_right, bottom_right, bottom_left])
    dst = np.float32([[offset, 0], [w-offset, 0], [w-offset, h], [offset, h]])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    binary_wraped = cv2.warpPerspective(combined, M, (w, h), flags=cv2.INTER_LINEAR)

    write_name= 'output_images/wrapped/wrapped_'+str(index+1)+'.jpg'
    print('wrapped image: ', write_name)
    cv2.imwrite(write_name, binary_wraped)

    #bottom_half_histogram(binary_wraped, index)
    
    #write_name= 'output_images/histogram/bottom_half_histogram_'+str(index+1)+'.jpg'
    #print('histogram image: ', write_name)
    #cv2.imwrite(write_name, half_histogram)

    sliding_window, left_fitx, right_fitx, ploty = fit_polynomial(binary_wraped, index)
    #write_name= 'output_images/sliding_window/sliding_window_fit_poly'+str(index+1)+'.jpg'
    #print('histogram image: ', write_name)
    #cv2.imwrite(write_name, sliding_window)

    left_curverad, right_curverad = measure_curvature_real(ploty, left_fitx, right_fitx)
    road_rad = (left_curverad+right_curverad)/2.0

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_wraped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)

    # calculate the offset of the car on the road
    camera_center = (left_fitx[-1] + right_fitx[-1])/2
    center_diff = (camera_center - w / 2) * xm_per_pix
    side_pos = 'left'
    if center_diff <= 0:
        side_pos = 'right'

    cv2.putText(result, 'Radius of Curvature = '+ str(round(road_rad, 3))+'(m)', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(result, 'Vehicle is '+str(abs(round(center_diff, 3))) + 'm ' + side_pos + ' of center', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2) 

    write_name= 'output_images/wrap_to_original/wrap_to_original'+str(index+1)+'.jpg'
    print('histogram image: ', write_name)
    cv2.imwrite(write_name, result)
