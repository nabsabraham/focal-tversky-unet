# -*- coding: utf-8 -*-
"""
@author: Nabilla Abraham
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

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
set_session(tf.Session(config=config))

img_row = 128
img_col = 128
img_size = 128
img_chan = 1
epochnum = 100
batchnum = 16 
input_size = (img_row, img_col, img_chan)
    
sgd = SGD(lr=0.01, momentum=0.9)

curr_dir = os.getcwd()
img_dir = os.path.join(curr_dir, 'original')
gt_dir = os.path.join(curr_dir, 'gt')

img_list = os.listdir(img_dir)
gt_list = os.listdir(gt_dir)

num_imgs = len(img_list)

orig_imgs = []
orig_gts = []
imgs = np.zeros((num_imgs, img_row, img_col))
gts = np.zeros_like(imgs)

for i in range(num_imgs):
    tmp_img = plt.imread(os.path.join(img_dir, img_list[i]))
    tmp_gt = plt.imread(os.path.join(gt_dir, img_list[i]))
    orig_imgs.append(tmp_img)
    orig_gts.append(tmp_gt)
    
    imgs[i] = cv2.resize(tmp_img, (img_col,img_row), interpolation=cv2.INTER_NEAREST)
    gts[i] = cv2.resize(tmp_gt,(img_col,img_row), interpolation=cv2.INTER_NEAREST)
    
indices = np.arange(0,num_imgs,1)

imgs_train, imgs_test, \
imgs_mask_train, orig_imgs_mask_test,\
trainIdx, testIdx = train_test_split(imgs,gts, indices,test_size=0.25)

imgs_train = np.expand_dims(imgs_train, axis=3)
imgs_mask_train = np.expand_dims(imgs_mask_train,axis=3)
imgs_test = np.expand_dims(imgs_test, axis=3)

filepath="weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_dsc', 
                             verbose=1, save_best_only=True, 
                             save_weights_only=True, mode='max')
gt1 = imgs_mask_train[:,::8,::8,:]
gt2 = imgs_mask_train[:,::4,::4,:]
gt3 = imgs_mask_train[:,::2,::2,:]
gt4 = imgs_mask_train
gt_train = [gt1,gt2,gt3,gt4]

model = newmodels.unet(sgd, input_size, losses.tversky_loss)
hist = model.fit(imgs_train, imgs_mask_train, validation_split=0.15,
                 shuffle=True, epochs=epochnum, batch_size=batchnum,
                 verbose=True, callbacks=[checkpoint])#, callbacks=[estop,tb])
h = hist.history
utils.plot(h, epochnum, batchnum, img_col, 0)

num_test = len(imgs_test)
_,_,_,preds = model.predict(imgs_test)
#preds = model.predict(imgs_test)

preds_up=[]
dsc = np.zeros((num_test,1))
recall = np.zeros_like(dsc)
tn = np.zeros_like(dsc)
prec = np.zeros_like(dsc)

thresh = 0.5

for i in range(num_test):
    gt = orig_gts[testIdx[i]]
    preds_up.append(cv2.resize(preds[i], (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST))
    dsc[i] = utils.check_preds(preds_up[i] > thresh, gt)
    recall[i], _, prec[i] = utils.auc(gt, preds_up[i] >thresh)
    
print('-'*30)
print('At threshold =', thresh)
print('\n DSC \t\t{0:^.3f} \n Recall \t{1:^.3f} \n Precision\t{2:^.3f}'.format(
        np.sum(dsc)/num_test,  
        np.sum(recall)/num_test,
        np.sum(prec)/num_test ))

model.load_weights("weights.hdf5")
_,_,_,preds = model.predict(imgs_test)
#preds = model.predict(imgs_test)   #use this if model is unet

preds_up=[]
dsc = np.zeros((num_test,1))
recall = np.zeros_like(dsc)
tn = np.zeros_like(dsc)
prec = np.zeros_like(dsc)

for i in range(num_test):
    gt = orig_gts[testIdx[i]]
    preds_up.append(cv2.resize(preds[i], (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST))
    dsc[i] = utils.check_preds(preds_up[i] > thresh, gt)
    recall[i], _, prec[i] = utils.auc(gt, preds_up[i] >thresh)
    
print('-'*30)
print('USING HDF5 MODEL', thresh)
print('\n DSC \t\t{0:^.3f} \n Recall \t{1:^.3f} \n Precision\t{2:^.3f}'.format(
        np.sum(dsc)/num_test,  
        np.sum(recall)/num_test,
        np.sum(prec)/num_test ))

# check to see how much accuracy we've lost by upsampling the predictions by comparing to 
# the original shapes used for training  
for i in range(num_test):
    gt = orig_imgs_mask_test[i]
    dsc[i] = utils.check_preds(np.squeeze(preds[i]) > thresh, gt)
    recall[i], _, prec[i] = utils.auc(gt, np.squeeze(preds[i]) >thresh)
    
print('-'*30)
print('Without resizing the preds =', thresh)
print('\n DSC \t\t{0:^.3f} \n Recall \t{1:^.3f} \n Precision\t{2:^.3f}'.format(
        np.sum(dsc)/num_test,  
        np.sum(recall)/num_test,
        np.sum(prec)/num_test ))

idx = np.random.randint(0,num_test)
gt_plot = orig_gts[testIdx[idx]]
plt.figure(dpi=200)
plt.subplot(121)
plt.imshow(np.squeeze(gt_plot), cmap='gray')
plt.title('Original Img {}'.format(idx))
plt.subplot(122)
plt.imshow(np.squeeze(preds_up[idx]), cmap='gray')
plt.title('Ground Truth {}'.format(idx))

y_true = orig_imgs_mask_test.ravel() 
y_preds = preds.ravel() 
precision, recall, thresholds = precision_recall_curve(y_true, y_preds)
plt.figure(20)
plt.plot(recall,precision)
