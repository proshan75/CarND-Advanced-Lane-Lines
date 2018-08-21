from moviepy.editor import VideoFileClip
from IPython.display import HTML
import numpy as np 
import cv2
import pickle
import pickle

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

def process_video_img(image):
    undistorted_image = cv2.undistort(image, cameraMatrix, distCoeffs, None, cameraMatrix)

    # Choose a Sobel kernel size
    ksize = 15 # Choose a larger odd number to smooth gradient measurements

    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(undistorted_image, orient='x', sobel_kernel=ksize, thresh_min=20, thresh_max=200)
    grady = abs_sobel_thresh(undistorted_image, orient='y', sobel_kernel=ksize, thresh_min=50, thresh_max=200)
    mag_binary = mag_thresh(undistorted_image, sobel_kernel=ksize, mag_thresh=(60, 130))
    dir_binary = dir_threshold(undistorted_image, sobel_kernel=ksize, thresh=(0.7, 1.3))
    c_binary = color_threshold(undistorted_image, sthreshold=(90,255), vthreshold=(90,255))

    combined = np.zeros_like(undistorted_image[:,:,0])
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (c_binary == 1) ] = 255 #

    h, w = undistorted_image.shape[:2]
    trap_top_width_per = 0.032
    trap_bot_width_per = 0.35
    trap_top_height_per = 0.62
    trap_bot_height_per = 0.98
    top_left = [w* (0.5 - trap_top_width_per) , h * trap_top_height_per]
    top_right = [w* (0.5 + trap_top_width_per), h * trap_top_height_per]
    bottom_left = [w* (0.5 - trap_bot_width_per), h * trap_bot_height_per]
    bottom_right = [w* (0.5 + trap_bot_width_per), h * trap_bot_height_per]
    offset = w * 0.25

    result = combined

    src = np.float32([top_left, top_right, bottom_right, bottom_left])
    dst = np.float32([[offset, 0], [w-offset, 0], [w-offset, h], [offset, h]])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    binary_wraped = cv2.warpPerspective(combined, M, (w, h), flags=cv2.INTER_LINEAR)

    #bottom_half_histogram(binary_wraped, index)

    sliding_window, left_fitx, right_fitx, ploty = fit_polynomial(binary_wraped)

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

    return result

Output_video = 'output_challenge_video.mp4'
Input_video = 'challenge_video.mp4'

clip1 = VideoFileClip(Input_video)
video_clip = clip1.fl_image(process_video_img)
video_clip.write_videofile(Output_video, audio=False)
