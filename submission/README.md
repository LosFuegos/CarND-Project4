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

[image1]: ./images/calibration2.jpg "distorted"
[image2]: ./images/undistorted_calibration2.jpg "undistorted"
[image3]: ./images/straight_lines1.jpg "test images"
[image4]: ./images/undistorted_straight_lines1.jpg "undistorted test image"
[image5]: ./images/binary.jpg "Binary Image"
[image6]: ./images/warped.jpg "Warped Image"
[image7]: ./images/projected.jpg "distorted"
[image7]: ./images/fit_lines.jpg "Fitted Lines"
[video1]: ./project_video_done.mp4 "Video"
[video2]: ./challenge_video_done.mp4 "Challenge"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  


###Camera Calibration

The code for this step is contained in "cal.py".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]
![alt text][image2]

###Pipeline (single images)

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image3]
![alt text][image4]
####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color thresholds to generate a binary image (thresholding steps at lines # through # in `image.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image5]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()`, which appears in lines 1 through 8 in the file `images.py`.  The `warp()` function takes the input image as (`img`), as well as source (`src`) and destination (`dst`) points and uses OpenCV's getPerspectiveTransform to get 'M' and 'Minv' fo.  I chose the hardcode the source and destination points in the following manner:

```
    src = np.float32([[544, 470], [736, 470],[128, 700], [1152, 700]])

    dst = np.float32([[256, 128], [1024, 128],[256, 720], [1024, 720]])
```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 544, 470      | 256, 128      | 
| 736, 470      | 1024, 128     |
| 128, 700      | 256, 720      |
| 1152, 700     | 1024, 720     |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appeared parallel in the warped image.

![alt text][image6]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In tracker.py lines 92-136 I experimented with margin, nwindows, and minpix variables and used the calculated left and right points from my sliding window search to  fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][imageIDK]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 146 through 159 in my code in `tracker.py`. Used the right and left points and the meter per pixel conversion to find the radii of curvatures. Used the difference the center of camera and warped image.

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 44 through 61 in my code in `video.py` in the function `project()`.  Here is an example of my result on a test image:

![alt text][image7]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further. My current pipeline might fail in more extreme curves and in more extreme lighting conditions like in the harder_challenge video. We could possibly overcome this with more robust thresholding and parameter selection. I tested out directional threshold on both challenge videos and did improve my performance but further work is needed.
---
