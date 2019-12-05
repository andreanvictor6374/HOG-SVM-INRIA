# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 20:53:18 2017

@author: Samyakh Tukra
"""
#%%
import matplotlib.pyplot as plt
from skimage import io
from skimage.feature import hog
from skimage import data, color, exposure
from PIL import Image
import cv2
import imutils
#%%
image_lists=['crop001706.png','crop001514.png','crop001573.png','Tamsui trip_191201_0017.jpg','Tamsui trip_191201_0120.jpg','Tamsui trip_191201_0126.jpg']
img_name=image_lists[5]
#img_name='victor1.jpg'
#img_name='crop001706.png'

#img = io.imread(r"Test_images/{}".format(img_name))

img= cv2.imread("Test_images/{}".format(img_name))
if img.shape[0]<img.shape[1]:
    img = imutils.resize(img, height=200)
else:
    img = imutils.resize(img, width=200)


image = color.rgb2gray(img)

fd, hog_image = hog(image, orientations=9, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualise=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input image')
#ax1.set_adjustable('box-forced')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
#ax1.set_adjustable('box-forced')
plt.savefig(img_name[:-4]+'_HOG.png')
plt.show()
