# Importing the necessary modules:

from skimage.feature import hog
from skimage.transform import pyramid_gaussian
from skimage.io import imread
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC,SVC
from sklearn.metrics import classification_report
#from sklearn.model_selection import train_test_split
from skimage import color
from imutils.object_detection import non_max_suppression
from sklearn.utils import shuffle
import imutils
import numpy as np
import os
import glob
from PIL import Image 
from numpy import *
full_train=False
saveModel=True
# define parameters of HOG feature extraction
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)
resize_const=200

#neg_im_path= r"../INRIAPerson/{}/neg".format('train_64x128_H96')
#pos_im_path= r"../INRIAPerson/{}/pos".format('train_64x128_H96')

def Extract_HOG_descriptor(mode='train_64x128_H96'):
    pos_im_path = r"../INRIAPerson/{}/pos".format(mode) 
    neg_im_path= r"../INRIAPerson/{}/neg".format(mode)
    # read the image files:
    pos_im_listing = os.listdir(pos_im_path) 
    neg_im_listing = os.listdir(neg_im_path)
#    print('num_pos_samples',size(pos_im_listing)) 
#    print('num_neg_samples',size(neg_im_listing))
    data= []
    labels = []

    for file in pos_im_listing: 
        for j in range(1):
            img = Image.open(pos_im_path + '/' + file)
            if j==0:
                img = img.resize((64,128))
            else:
                # croped image in the center 
                center=img.height//2,img.width//2
                left = center[1]-32
                right = center[1]+32
                top = center[0]-64
                bottom = center[0]+64
                img = img.crop((left, top, right, bottom))
                
            gray = img.convert('L') 
            # calculate HOG for positive features
            fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True)# fd= feature descriptor
            data.append(fd)
            labels.append(1)
        
    for file in neg_im_listing:
        for j in range(2): 
            img= Image.open(neg_im_path + '/' + file)
            if j==0:
                img = img.resize((64,128))
            else:
                ratio=img.height/img.width
                img = img.resize((resize_const,int(resize_const*ratio)))
                if j==1:
                    center=img.height//2,img.width//2
                    left = center[1]-32
                    right = center[1]+32
                elif j==2:
                    left = img.width-64
                    right = img.width

                elif j==3:
                    left = 0
                    right = 64
                top = center[0]-64
                bottom = center[0]+64
                img = img.crop((left, top, right, bottom))

            gray= img.convert('L')
            # Now we calculate the HOG for negative features
            fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True) 
            data.append(fd)
            labels.append(0)
    
    # encode the labels, converting them from strings to integers
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    return np.array(data),np.array(labels)

#%%================Partitioning data=============================================
print(" Constructing training/testing split...")
trainData,trainLabels=Extract_HOG_descriptor()
testData,testLabels=Extract_HOG_descriptor(mode='test_64x128_H96')
trainData, trainLabels = shuffle(trainData, trainLabels)
testData, testLabels = shuffle(testData, testLabels)
#(trainData, testData, trainLabels, testLabels) = train_test_split(
#	np.array(trainData), trainLabels, test_size=0.20, random_state=42)
print('# of training Data:',trainData.shape[0],';# of negative label:',sum(trainLabels==0),';# of positive label:',sum(trainLabels==1)) 
print('# of testing Data:',testData.shape[0],';# of negative label:',sum(testLabels==0),';# of positive label:',sum(testLabels==1)) 
#%% Train the linear SVM
print(" Training SVM classifier...")
#CHOOSE SVC() OR LinearSVC()
#model = SVC(C=1,gamma=(1/3780)) 
model = SVC(C=1,gamma=(1/(3780*trainData.std()**2)))
#model = SVC(C=0.01,gamma=0.1)   
#model = LinearSVC(C=0.01)
model.fit(trainData, trainLabels)
#%% Evaluate the classifier
print(" Evaluating classifier on test data ...")
predictions = model.predict(testData)
print(classification_report(testLabels, predictions))

#%%===========training on entire dataset we have===============================
if full_train:
    DATA=np.append(trainData,testData,axis=0)
    LABELS=np.append(trainLabels,testLabels,axis=0)
    model_full = LinearSVC()
    model_full.fit(DATA, LABELS)

# Save the model:
#%% Save the Model
if saveModel:
    try:
      joblib.dump(model_full, 'HOG_SVM_model.npy')
      print("model_full is successfully saved")
    except:
      joblib.dump(model, 'HOG_SVM_model.npy')
      print("model is successfully saved")
    
    
