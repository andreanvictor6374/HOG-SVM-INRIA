from skimage.feature import hog
from skimage.transform import pyramid_gaussian
from sklearn.externals import joblib
from skimage import color
from imutils.object_detection import non_max_suppression
import imutils
import numpy as np
import cv2
import os
import glob
import matplotlib.pyplot as plt
#================preferences=================================================
image_lists=['crop001706.png','crop001514.png','crop001573.png','Tamsui trip_191201_0017.jpg','Tamsui trip_191201_0120.jpg','Tamsui trip_191201_0126.jpg']
img_name=image_lists[0]

minAxis_Image=200

#Define HOG Parameters
# change them if necessary to orientations = 8, pixels per cell = (16,16), cells per block to (1,1) for weaker HOG
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)
threshold = 0.6

# define the sliding window:
def sliding_window(image, stepSize, windowSize):# image is the input, step size is the no.of pixels needed to skip and windowSize is the size of the actual window
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):# this line and the line below actually defines the sliding part and loops over the x and y coordinates
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y: y + windowSize[1], x:x + windowSize[0]])
#%%
# Upload the saved svm model:
model = joblib.load('HOG_SVM_model.npy')

# Test the trained classifier on an image below!
scale = 0
detections = []
img_ori= cv2.imread("Test_images/{}".format(img_name))

 
if img_ori.shape[0]<img_ori.shape[1]:
    img = imutils.resize(img_ori, height=minAxis_Image)
else:
    img = imutils.resize(img_ori, width=minAxis_Image)

HO,WO,_=img_ori.shape
HR,WR,_=img.shape

A = np.copy(img)

# defining the size of the sliding window (has to be, same as the size of the image in the training data)
(winW, winH)= (64,128)
windowSize=(winW,winH)
downscale=1.2
# Apply sliding window:
for resized in pyramid_gaussian(img, downscale=downscale): # loop over each layer of the image that you take!
    # loop over the sliding window for each layer of the pyramid
    for (x,y,window) in sliding_window(resized, stepSize=5, windowSize=(winW,winH)):
        # if the window does not meet our desired window size, ignore it!
        if window.shape[0] != winH or window.shape[1] !=winW: # ensure the sliding window has met the minimum size requirement
            continue
        window=color.rgb2gray(window)
        fds = hog(window, orientations, pixels_per_cell, cells_per_block, block_norm='L2')  # extract HOG features from the window captured
        fds = fds.reshape(1, -1) # re shape the image to make a silouhette of hog
        pred = model.predict(fds) # use the SVM model to make a prediction on the HOG features extracted from the window
        
        if pred == 1:
            if model.decision_function(fds) > threshold:  # set a threshold value for the SVM prediction i.e. only firm the predictions above probability of 0.8
                print("imsize",resized.shape)
                print("Detection:: Location -> ({}, {})".format(x, y))
                print("Scale ->  {} | Confidence Score {} \n".format(scale,model.decision_function(fds)))
                
                detections.append((int(x * (downscale**scale)), int(y * (downscale**scale)), model.decision_function(fds),
                                   int(windowSize[0]*(downscale**scale)), # create a list of all the predictions found
                                      int(windowSize[1]*(downscale**scale))))
    scale+=1
    
clone = resized.copy()
for (x_tl, y_tl, _, w, h) in detections:
    cv2.rectangle(img, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 0, 255), thickness = 2)
rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections]) # do nms on the detected bounding boxes
sc = [score[0] for (x, y, score, w, h) in detections]
print("detection confidence score: ", sc)
sc = np.array(sc)
pick = non_max_suppression(rects, probs = sc, overlapThresh = 0.3)

for (xA, yA, xB, yB) in pick:
    cv2.rectangle(img_ori, (int(xA*(WO/WR)), int(yA*(HO/HR))), (int(xB*(WO/WR)), int(yB*(HO/HR))), (0,255,0), 2)
for (xA, yA, xB, yB) in pick:
    cv2.rectangle(A, (xA, yA),(xB, yB),(0,255,0), 1)
#cv2.imshow("Raw Detections after NMS", A)
plt.imshow(img)
plt.show()
#plt.imshow(A)
#plt.show()
plt.imshow(img_ori)
cv2.imwrite(img_name[:-4]+'_high_resolution.png',img_ori)
#cv2.imwrite(img_name+'_low_resulution.png',A)
cv2.imwrite(img_name[:-4]+'_before_NMS.png',img)

