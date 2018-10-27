# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 15:12:49 2018

@author: Nabila Abraham
"""
import numpy as np
import matplotlib.pyplot as plt 


def plot(hist, epochnum, batchnum, name, is_attnnet=0):
    plt.figure()

    if is_attnnet==True:
        train_loss = hist['final_loss']
        val_loss = hist['val_final_loss']
        acc = hist['final_dsc'] 
        val_acc = hist['val_final_dsc']
    else:
        train_loss = hist['loss']
        val_loss = hist['val_loss']
        acc = hist['dsc'] 
        val_acc = hist['val_dsc']
        
    epochs = np.arange(1, len(train_loss)+1,1)
    plt.plot(epochs,train_loss, 'b', label='Training Loss')
    plt.plot(epochs,val_loss, 'r', label='Validation Loss')
    plt.grid(color='gray', linestyle='--')
    plt.legend()
    plt.title('LOSS Model={}, Epochs={}, Batch={}'.format(name,epochnum, batchnum))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.figure()
    plt.plot(epochs, acc, 'b', label='Training Dice Coefficient')
    plt.plot(epochs, val_acc, 'r', label='Validation Dice Coefficient')
    plt.grid(color='gray', linestyle='--')
    plt.legend()            
    plt.title('DSC Model={}, Epochs={}, Batch={}'.format(name,epochnum, batchnum))
    plt.xlabel('Epochs')
    plt.ylabel('Dice')


def check_preds(ypred, ytrue):
    smooth = 1
    pred = np.ndarray.flatten(np.clip(ypred,0,1))
    gt = np.ndarray.flatten(np.clip(ytrue,0,1))
    intersection = np.sum(pred * gt) 
    union = np.sum(pred) + np.sum(gt)   
    return np.round((2 * intersection + smooth)/(union + smooth),decimals=5)

def confusion(y_true, y_pred):
    smooth = 1
    y_pred_pos = np.round(np.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = np.round(np.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    tp = (np.sum(y_pos * y_pred_pos) + smooth) / (np.sum(y_pos) + smooth) 
    tn = (np.sum(y_neg * y_pred_neg) + smooth) / (np.sum(y_neg) + smooth)
    return [tp, tn]

def auc(y_true, y_pred):
    smooth = 1
    y_pred_pos = np.round(np.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = np.round(np.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    tp = np.sum(y_pos * y_pred_pos)
    tn = np.sum(y_neg * y_pred_neg)
    fp = np.sum(y_neg * y_pred_pos)
    fn = np.sum(y_pos * y_pred_neg)
    tpr = (tp + smooth) / (tp + fn + smooth) #recall
    tnr = (tn + smooth) / (tn + fp + smooth)
    prec = (tp + smooth) / (tp + fp + smooth) #precision
    return [tpr, tnr, prec]