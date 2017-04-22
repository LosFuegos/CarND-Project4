import cv2
import pickle
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tracker import Line

dict_pickle = pickle.load(open("./calibration.p", "rb"))
mtx = dict_pickle['mtx']
dist = dict_pickle['dist']

#attemped other thresholding techniques but settled on color threshold
def Threshold(img, thresh=(170,255)):
    binary = np.zeros_like(img)
    binary[(img >= thresh[0]) & (img <= thresh[1])] = 1
    return binary
def dir_threshold(gray, sobel_kernel=15, thresh=(0.7, 1.4)):

    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # 3) Take the absolute value of the x and y gradients
    abs_s_x = np.absolute(sobelx)
    abs_s_y = np.absolute(sobely)

    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction
    # of the gradient
    # Important, y should come before x here if we want to detect lines
    dir_grad = np.arctan2(abs_s_y, abs_s_x)

    # 5) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(dir_grad)

    # 6) Return this mask as your binary_output image
    binary_output[(dir_grad >= thresh[0]) & (dir_grad <= thresh[1])] = 1

    return binary_output
def combine(*arrs):
    combined = np.zeros_like(arrs[0])
    for img in arrs:
        combined[(combined == 1) & (img == 1)] = 255
    return combined

def warp(img):
    size = (img.shape[1], img.shape[0])

    src = np.float32([[246,678],[1052,678],[702,460],[575,460]])

    dst = np.float32([[320,720],[980,720],[980,0],[320,0]])
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

def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

def draw(window_centroids, warped, window_height, window_width):
    l_points = np.zeros_like(warped)
    r_points = np.zeros_like(warped)
    rightx = []
    leftx = []
    # Go through each level and draw the windows
    for level in range(0,len(window_centroids)):
        # Window_mask is a function to draw window areas
        leftx.append(window_centroids[level][0])
        rightx.append(window_centroids[level][1])
        l_mask = window_mask(window_width,window_height,warped,window_centroids[level][0],level)
        r_mask = window_mask(window_width,window_height,warped,window_centroids[level][1],level)
        # Add graphic points from window mask here to total pixels found
        l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
        r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

    template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
    zero_channel = np.zeros_like(template) # create a zero color channel
    template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
    warpage = np.array(cv2.merge((warped,warped,warped)),np.uint8) # making the original road pixels 3 color channels
    output = cv2.addWeighted(warpage, 1, template, 1.5, 0.0) # overlay the orignal road image with window results

    return output, leftx, rightx

def pipeline(img, mtx, dist, curves):
    #undistort images
    img = cv2.undistort(img, mtx, dist, None, mtx)

    #Converting to different Colorspaces
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    #Isolating Color Spaces
    l = lab[:,:,0]
    b = lab[:,:,2]
    v = hsv[:,:,2]
    r = img[:,:,0]
    u = yuv[:,:,1]

    #applying thresholding
    dir_t = dir_threshold(gray)
    bi_V = Threshold(v, thresh=(229,255))
    bi_R = Threshold(r,thresh=(234,255))
    bi_U = Threshold(u,thresh=(145,255))
    bi_l = Threshold(l,thresh=(210,255))
    bi_b = Threshold(b,thresh=(145,255))

    #combining 5 thresholded images
    combined = combine(bi_V, bi_R, bi_U, bi_l, bi_b, dir_t)
    #performing perspective transform
    warped, Minv = warp(combined)

    #finding lane lines

    #centroids = curves.find_window_centroids(warped)

    #drawn, leftx, rightx = draw(centroids, warped, height, width)

    #left_fitx, right_fitx, ploty, curves.left_curverad, curves.right_curverad = getCurvature(warped, curves.height, curves.width, leftx, rightx)
    #result = project(img, warped, Minv, ploty, left_fitx, right_fitx)

    plt.imshow(combined)
    plt.show()

    return combined

#files names for images in test_images folder
testImgs = ['straight_lines1.jpg', 'straight_lines2.jpg', 'test1.jpg', 'test2.jpg',
           'test3.jpg', 'test4.jpg', 'test5.jpg', 'test6.jpg']

#folders where input images and video are stored
folder2 = 'test_images/'

#output folder for all outputed images or video
outfolder = 'output_images/'

width = 100
height = 40
margin = 25
curves = Line(width, height, margin, ym=10/720, xm=4/384, smoothing=15)
for file in testImgs:
    path = folder2 + file
    img = mpimg.imread(path)
    save = outfolder + 'undistorted_' + file
    result = pipeline(img, mtx, dist, curves)
    plt.imsave(save, result)