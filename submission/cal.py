import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def Calibrate(Images,folder, nx=9, ny=6):
    imgpoints = [] #3D points in Real World Space
    objpoints = [] #2D points in image plane
    objp = np.zeros((ny*nx, 3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2) # x and y
    for file in Images:
        img = cv2.imread(folder + file)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)
            #img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)

    return imgpoints, objpoints

def undistort(Images, folder, outfolder, objpoints, imgpoints):
    outname = ''
    for file in Images:
        img = cv2.imread(folder + file)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[0:2], None, None)
        undist = cv2.undistort(img, mtx, dist, None, mtx)
        outname = outfolder + 'undistorted_' + file.split('/')[-1]
        cv2.imwrite(outname, undist)
    return mtx, dist
#File names for images in camera_cal folder
Images = ['calibration1.jpg', 'calibration2.jpg', 'calibration3.jpg', 'calibration4.jpg',
          'calibration5.jpg', 'calibration6.jpg', 'calibration7.jpg', 'calibration8.jpg',
          'calibration9.jpg','calibration10.jpg', 'calibration11.jpg','calibration12.jpg',
          'calibration13.jpg','calibration14.jpg','calibration15.jpg','calibration16.jpg',
          'calibration17.jpg','calibration18.jpg','calibration19.jpg','calibration20.jpg']

#folders where input images and video are stored
folder1 = 'camera_cal/'

#output folder for all outputed images or video
outfolder = 'output_images/'

#finding chessboard corners and points
imgpoints, objpoints = Calibrate(Images, folder1)

#Undistorts images in camera_cal folder
mtx , dist = undistort(Images, folder1, outfolder, objpoints, imgpoints)

dist_pickle = {}

dist_pickle['mtx'] = mtx

dist_pickle['dist'] = dist

pickle.dump(dist_pickle, open('./calibration.p', "wb"))