# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 17:16:54 2018

@author: Nabila Abraham
"""

import os 
import cv2 
import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt 

from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers import Activation, add, multiply, Lambda
from keras.layers import AveragePooling2D, average, UpSampling2D, Dropout
from keras.optimizers import Adam, SGD, RMSprop
from keras.initializers import glorot_normal, random_normal, random_uniform
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras import backend as K
from keras.layers.normalization import BatchNormalization 

from sklearn.metrics import roc_curve, auc, precision_recall_curve # roc curve tools
from sklearn.model_selection import train_test_split

import losses 
import utils 
import newmodels

img_row = 192
img_col = 256
img_chan = 3
epochnum = 50
batchnum = 16  
smooth = 1.
input_size = (img_row, img_col, img_chan)
    
sgd = SGD(lr=0.01, momentum=0.90, decay=1e-6)
adam = Adam(lr=1e-3) 

curr_dir = os.getcwd()
train_dir = os.path.join(curr_dir, 'resized_train')
gt_dir = os.path.join(curr_dir, 'resized_gt')
orig_dir = os.path.join(curr_dir, 'orig_gt')

img_list = os.listdir(train_dir)
num_imgs = len(img_list)

orig_data = np.zeros((num_imgs, img_row, img_col, img_chan))
orig_masks = np.zeros((num_imgs, img_row, img_col,1))

for idx,img_name in enumerate(img_list): 
    orig_data[idx] = plt.imread(os.path.join(train_dir, img_name))
    orig_masks[idx,:,:,0] =  plt.imread(os.path.join(gt_dir, img_name.split('.')[0] + "_segmentation.png"))
   
indices = np.arange(0,num_imgs,1)

imgs_train, imgs_test, \
imgs_mask_train, orig_imgs_mask_test,\
trainIdx, testIdx = train_test_split(orig_data,orig_masks, indices,test_size=0.25)

imgs_train /= 255
imgs_test /=255

estop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, mode='auto')
filepath="weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_final_dsc', 
                             verbose=1, save_best_only=True, 
                             save_weights_only=True, mode='max')
gt1 = imgs_mask_train[:,::8,::8,:]
gt2 = imgs_mask_train[:,::4,::4,:]
gt3 = imgs_mask_train[:,::2,::2,:]
gt4 = imgs_mask_train
gt_train = [gt1,gt2,gt3,gt4]

model = newmodels.attn_reg(sgd, input_size, losses.focal_tversky)
hist = model.fit(imgs_train, gt_train, validation_split=0.15,
                 shuffle=True, epochs=epochnum, batch_size=batchnum,
                 verbose=True, callbacks=[checkpoint])#, callbacks=[estop,tb])
h = hist.history
utils.plot(h, epochnum, batchnum, img_col, 1)

num_test = len(imgs_test)
_,_,_,preds = model.predict(imgs_test)
#preds = model.predict(imgs_test)   #use this if the model is unet

preds_up=[]
dsc = np.zeros((num_test,1))
recall = np.zeros_like(dsc)
tn = np.zeros_like(dsc)
prec = np.zeros_like(dsc)

thresh = 0.5

# check the predictions from the trained model 
for i in range(num_test):
    #gt = orig_masks[testIdx[i]]
    name = img_list[testIdx[i]]
    gt = plt.imread(os.path.join(orig_dir, name.split('.')[0] + "_segmentation.png")) 

    pred_up = cv2.resize(preds[i], (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
    dsc[i] = utils.check_preds(pred_up > thresh, gt)
    recall[i], _, prec[i] = utils.auc(gt, pred_up >thresh)
    
print('-'*30)
print('At threshold =', thresh)
print('\n DSC \t\t{0:^.3f} \n Recall \t{1:^.3f} \n Precision\t{2:^.3f}'.format(
        np.sum(dsc)/num_test,  
        np.sum(recall)/num_test,
        np.sum(prec)/num_test ))

# check the predictions with the best saved model from checkpoint
model.load_weights("weights.hdf5")
_,_,_,preds = model.predict(imgs_test)
#preds = model.predict(imgs_test)   #use this if the model is unet

preds_up=[]
dsc = np.zeros((num_test,1))
recall = np.zeros_like(dsc)
tn = np.zeros_like(dsc)
prec = np.zeros_like(dsc)

for i in range(num_test):
    #gt = orig_masks[testIdx[i]]
    name = img_list[testIdx[i]]
    gt = plt.imread(os.path.join(orig_dir, name.split('.')[0] + "_segmentation.png")) 

    pred_up = cv2.resize(preds[i], (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
    dsc[i] = utils.check_preds(pred_up > thresh, gt)
    recall[i], _, prec[i] = utils.auc(gt, pred_up >thresh)
    
print('-'*30)
print('USING HDF5 saved model at thresh=', thresh)
print('\n DSC \t\t{0:^.3f} \n Recall \t{1:^.3f} \n Precision\t{2:^.3f}'.format(
        np.sum(dsc)/num_test,  
        np.sum(recall)/num_test,
        np.sum(prec)/num_test ))
    
#plot precision-recall 
y_true = orig_imgs_mask_test.ravel() 
y_preds = preds.ravel() 
precision, recall, thresholds = precision_recall_curve(y_true, y_preds)
plt.figure(20)
plt.plot(recall,precision)

