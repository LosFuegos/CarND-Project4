import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from tracker import Line

dict_pickle = pickle.load(open("./calibration.p", "rb"))
mtx = dict_pickle['mtx']
dist = dict_pickle['dist']

#attemped other thresholding techniques but settled on color threshold
def Threshold(img, thresh=(170,255)):
    binary = np.zeros_like(img)
    binary[(img >= thresh[0]) & (img <= thresh[1])] = 1
    return binary

def dir_threshold(img, sobel_kernel=31, thresh=(0, np.pi/2)):
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

def combine(*arrs):
    combined = np.zeros_like(arrs[0])
    for img in arrs:
        combined[(combined == 1) | (img == 1)] = 1
    return combined

def warp(img):
    size = (img.shape[1], img.shape[0])

    src = np.float32([[544, 470], [736, 470],[128, 700], [1152, 700]])

    dst = np.float32([[256, 128], [1024, 128],[256, 720], [1024, 720]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, size, flags=cv2.INTER_LINEAR)

    return warped, Minv

def project(img, warped, Minv, ploty, left_fitx, right_fitx):
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

    return result

def getCurvature(lefty, righty, ploty, warped, leftx, rightx, left_fitx, right_fitx, left_fit, right_fit):

    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    #print(left_curverad, right_curverad)

    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    #print(left_curverad, 'm', right_curverad, 'm')
    # Example values: 632.1 m    626.2 m

    camera_center = (left_fitx[-1] + right_fitx[-1])/2
    center_diff = (camera_center-warped.shape[1]/2) * xm_per_pix
    side_pos = 'left'
    if center_diff <= 0:
        side_pos = 'right'

    return left_curverad, right_curverad, center_diff, side_pos

def pipeline(img):
    #undistort images

    img = cv2.undistort(img, mtx, dist, None, mtx)

    #Converting to different Colorspaces
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    #Isolating Color Channel
    l = lab[:,:,0]
    v = hsv[:,:,2]
    r = img[:,:,0]
    u = yuv[:,:,1]
    b = lab[:,:,2]

    #applying thresholding
    #dirthresh = dir_threshold(gray, thresh=(.8, 1.))
    bi_V = Threshold(v, thresh=(229,255))
    bi_R = Threshold(r,thresh=(234,255))
    bi_U = Threshold(u,thresh=(145,255))
    bi_l = Threshold(l,thresh=(200,255))
    bi_b = Threshold(b,thresh=(145,255))


    #combining 5 thresholded images
    combined = combine(bi_V, bi_R, bi_U, bi_l, bi_b)    #performing perspective transform
    warped, Minv = warp(combined)

    curves = Line(margin=100, ym=10/720, xm=4/384, smoothing=60)

    #finding lane lines

    left_curverad, right_curverad, center_diff, side_pos, left_fitx, right_fitx, ploty = curves.slidingWindow(warped)

    result = project(img, warped, Minv, ploty, left_fitx, right_fitx)

    cv2.putText(result, 'Radius of Left Curvature = ' + str(round(left_curverad,3))+'(m)',(50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2)
    cv2.putText(result, 'Radius of Right Curvature = ' + str(round(right_curverad,3))+'(m)',(50,100), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2)
    cv2.putText(result, 'Vehicle is '+str(abs(round(center_diff,3)))+'m '+side_pos+' center of',(50,150),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)

    return result

level1 = 'project_video.mp4'
level2 = 'challenge_video.mp4'
level3 = 'harder_challenge_video.mp4'
mine = 'myOwnVideo.mp4'

level1_output = 'project_video_done.mp4'
level2_output = 'challenge_video_done.mp4'
level3_output = 'harder_challenge_video_done.mp4'
mine_output = 'myOwnVideo_done.mp4'
clip1 = VideoFileClip(level1)
white_clip = clip1.fl_image(pipeline)
white_clip.write_videofile(level1_output, audio=False)