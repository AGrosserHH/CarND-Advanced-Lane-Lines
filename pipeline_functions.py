import cv2
import numpy as np

# undistort image
def undistort(img, mtx, dist):
    return cv2.undistort(img, mtx, dist, None, mtx)

# Binary image
def binary(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HLS color space and separate the S channel
    # Note: img is the undistorted image
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]

    # Grayscale image
    # NOTE: we already saw that standard grayscaling lost color information for the lane lines
    # Explore gradients in other colors spaces / color channels to see what might work better
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    thresh_min = 20
    thresh_max = 100
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Threshold color channel
    s_thresh_min = 170
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
    
    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return combined_binary

# unwarp image
def corners_unwarp(img, mtx, dist):
    #Use the OpenCV undistort() function to remove distortion
    #undist = cv2.undistort(img, mtx, dist, None, mtx)        
    
    h,w = img.shape[:2]  
    
    src = np.float32([[w,h-10], #rd
                     [540,470], #lu
                     [740,470], #ru
                     [0, h-10]]) #ld
    dst= np.float32([[w,h], #rd
                     [0,0], #lu
                     [w,0], #ru
                     [0, h]]) #ld
    
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    
    warped = cv2.warpPerspective(img, M, (w,h), flags=cv2.INTER_LINEAR)
    
    return warped, M, Minv

#Definition of pipeline
def pipeline_img(img1, mtx, dist):
    
    # undistort image
    undist = undistort(img1, mtx, dist)
    
    # create binary
    binary_img = binary(undist, s_thresh=(170, 255), sx_thresh=(20, 100))
    
    # birdeye
    img_unwarped, M, Minv = corners_unwarp(binary_img, mtx, dist)
    
    return img_unwarped, M, Minv

