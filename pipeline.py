import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from pipeline_functions import *

dist_pickle = pickle.load( open( "camera_cal/wide_dist_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# Test on image
__img = mpimg.imread('test_images/straight_lines1.jpg')
binary_warped, M, Minv= pipeline_img(__img, mtx, dist)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()

ax1.imshow(__img)
ax1.set_title('Original Image', fontsize=40)

ax2.imshow(binary_warped)
ax2.set_title('Warped', fontsize=40)

print("success")
