import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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

def Threshold(img, thresh=(170,255)):
    binary = np.zeros_like(img)
    binary[(img >= thresh[0]) & (img <= thresh[1])] = 1
    return binary

def combine(*images):
    combined = np.zeros_like(images[0])
    for img in images:
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
def pipeline(img):
    #trying out different Color Channels
    #img = cv2.undistort(img, mtx, dist, None, mtx)

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

    dirthresh = dir_threshold(gray, thresh=(.79, 1.0))
    bi_V = Threshold(v, thresh=(225,255))
    bi_R = Threshold(r,thresh=(234,255))
    bi_U = Threshold(u,thresh=(145,255))
    bi_l = Threshold(l,thresh=(225,255))
    bi_b = Threshold(b,thresh=(145,253))

    #images = [bi_V, bi_R, bi_U, bi_l, bi_b, dirthresh]

    result = combine(bi_V, bi_R, bi_U, bi_l)
    #result = combine(bi_R, absX)
    #result = combine(absX, dirthresh)
    #result = dirthresh
    warped, Minv = warp(result)
    #project(warped)
    return result
images = glob.glob('./test_images/test*.jpg')

for idx, fname in enumerate(images):
    img = mpimg.imread(fname)
    print(fname)
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    result = pipeline(img)
    outfile = './findBest/' + str(idx) + '.jpg'
    plt.imsave(outfile, result, cmap='gray')
    plt.imshow(result, cmap='gray')
    plt.show()
    cv2.waitKey(0)
    print(idx)
