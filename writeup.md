## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./output_images/threshold/multi_thresholded_1.jpg "Threshold Binary image"
[image11]: ./output_images/threshold/multi_thresholded_2.jpg "Threshold Binary image"
[image4]: ./output_images/undistorted/undistorted_2.jpg "Undistorted"
[image12]: ./output_images/wrapped/wrapped_2.jpg "Wrapped"
[image13]: ./output_images/undistorted/undistorted_3.jpg "Undistorted"
[image14]: ./output_images/wrapped/wrapped_3.jpg "Wrapped"
[image5]: ./output_images/sliding_window/sliding_window_fit_poly3.jpg "Sliding Window Fit Polynomial Visual"
[image6]: ./output_images/wrap_to_original/wrap_to_original5.jpg "Wrapped to Original"
[video1]: ./output_project_video.mp4 "Video"
[image7]: ./output_images/corners_found9.jpg "Corners identified"
[image8]: ./camera_cal/calibration2.jpg "Original Image"
[image9]: ./output_images/chess_undistort/undistort_camera_cal12.jpg "Undistorted Image"
[image10]: ./output_images/undistorted/undistorted_1.jpg "Undistorted Image"
[image15]: ./output_images/histogram/bottom_half_histogram_3.jpg "Histogram"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

All functions needed for Camera Calibration is coded in `camera_calibration.py`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. Following image shows one of the chessboard image where the corners were detected successfully.

![alt text][image7]

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  

Here is an original image for which distortion correction is applied next.

![alt text][image8]

I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image9]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

When the `cv2.calibrateCamera()` function executed on the chessboard example images, we got the camera calibration and distortion coefficients values. They are stored in the pickle file `calibration_pickle.p`. The `test_images.py` file retrieves the two values at the beginning. Then loops through the images in `test_images` folder and calls `undestort` method to apply the distortion correction. Following image shows the corrected image to the one shown above:
![alt text][image10]


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I implemented various thresholding functions in `Threshold.py` file. Using `abs_sobel_thresh` function to evaluate gradients in x and y direction which will identify edges in vertical and horizontal direction respectively. The `mag_thresh` function evaluates the gradient magnitude in x and y orientation and generates the binary mask image. Followed by the `dir_threshold` function performs gradient direction calculation within the given range. Finally, the `color_threshold` function utilizes HSV as well as HLS color space to extract value channel and saturation channel respectively. These two channels are used to highlight features in the supplied image.

By combining various color and gradient thresholds mentioned earlier I generated a binary image. Here's an example of my output for this step.  

![alt text][image3]

Another example of threshold binary output.

![alt text][image11]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

I performed the perspective transform for test images in `test_images.py` file. Similar logic is implemented in `process_video_img` function of `test_video.py` which is meant to handle each image in the video stream.

The `process_video_img()` function defines a trapezoid shape as `src` and reshapes it to a rectangle in the `dst` (from line #36 to #45 of the source code).  

  Using the following approach I have parameterized the sizes of trapezoid which somewhat matches with the road in the perspective view:

  ```python
  h, w = image.shape[:2]
  trap_top_width_per = 0.032
  trap_bot_width_per = 0.35
  trap_top_height_per = 0.62
  trap_bot_height_per = 0.98
  ```

  Above parameters are used to define the trapezoid as below:

  ```python
  top_left = [w* (0.5 - trap_top_width_per) , h * trap_top_height_per]
  top_right = [w* (0.5 + trap_top_width_per), h * trap_top_height_per]
  bottom_left = [w* (0.5 - trap_bot_width_per), h * trap_bot_height_per]
  bottom_right = [w* (0.5 + trap_bot_width_per), h * trap_bot_height_per]
  offset = w * 0.25
  ```

  Finally the source and destination points are defined as follows:

  ```python
  src = np.float32([top_left,
                    top_right,
                    bottom_right,
                    bottom_left])
  dst = np.float32([[offset, 0],
                    [w-offset, 0],
                    [w-offset, h],
                    [offset, h]])
  ```

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 599.039, 446.399      | 320, 0        |
| 680.96, 446.399      | 960, 0      |
| 1088.0, 705.599     | 960, 720      |
| 192.0, 705.599      | 320, 720        |

I observed that my perspective transform was working as expected by comparing the image with road lanes in the undistorted test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

Corresponding wrapped image is:
![alt text][image12]

Another undistorted image:

![alt text][image13]

And corresponding wrapped image is:

![alt text][image14]


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

To identify the lane line pixel, I gathered histogram data for the bottom half-image. The `bottom_half_histogram` function in `sliding_window.py` file extract the data for input binary wrapped image. Here is an example of the histogram image captured:

![alt text][image15]

In most cases, the histogram peaks indicates the left and right side of the road lanes and helps to identify the lane starting points. Then using the sliding window approach implemented as part of the `find_lane_pixels` function in `sliding_window.py`, the lane pixels are identified scrolling from bottom to top. I used few hyperparameters such as number of windows (`nwindows`) , margin and minimum number of lane pixels (`minpix`) to identify the lane pixels and recenter the window.

Next the `fit_polynomial` function is used to fit a second order polynomial through the lane points identified. Following image shows the result of sliding window and polynomial line identified.

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The `measure_curvature_real` function in `measure_curve.py` implements the logic to measure the curvature radius. It also converts the pixel values to real world unit (m).

Using the following formula the radius of curvature is determined.
```
          (1+(2Ay+B)2)3/2​
Rcurve ​= -----------------
               ∣2A∣
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

As part of processing the test images in `test_images.py` file I implemented this step in lines #85 through #106 in my code.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The pipeline setup seems to work fairly well for the road profile recorded project_video. This road is mostly uniform with good lane marking, limited or no shade changes in the road near to the lanes. Due to that reason, the lane pixels are identified well and as seen the video output the lane detection is performed well.

Now when I tried the same with the challenging road video it fails few times. That road  seems to be constructed with two different concrete slabs. That construction profile seems to break the logic. Also, at one point the lane doesn't seem to be parallel and appears reducing in the width causing the lane capturing failure.
