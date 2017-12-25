#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 09:31:15 2017

@author: caiweiyun
"""

import numpy as np

batch_size = 200
lr = 0.01
lamda = 0.1
data_dir = '/Users/caiweiyun/Downloads/cifar-10-batches-py/data_batch_1'

W = {}
W['W1'] = np.random.randn(1000, 3072)/np.square(3072)
W['W2'] = np.random.randn(10, 1000)/np.square(3072)

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def inportdata(data_dir):
    data = unpickle(data_dir) 
    feature = data[b'data'] ##data is cifar-10 data from http://www.cs.toronto.edu/~kriz/cifar.html
    feature.astype(np.float64)
    label = data[b'labels']
    train_x = feature[:7999, :]
    train_y = label[0:7999]
    validation_x = feature[8000:8999, :]
    validation_y = label[8000:8999]
    test_x = feature[9000:9999, :]
    test_y = label[9000:9999]
    return train_x, train_y, validation_x, validation_y, test_x, test_y

def forward_pass(W,x):
    A = {}
    A['A1'] = Sigmoid(W['W1'].dot(x.T))
    A['A2'] = W['W2'].dot(A['A1'])
    OP = softmax(A['A2'])
    return A, OP

def Loss(OP,y):
    y_mat = np.zeros((10,))
    y_mat[y] = 1
#    loss = -np.sum((y_mat*np.log(OP)) + (1- y_mat)*np.log(1 - OP))/100
    loss = -np.sum((y_mat*np.log(OP)))
    return loss

def Backword(OP, A, W, x, y):
    """
    -OP one Output of the network
    -A  Activitions of the network
    -W  W matrix
    -x  one input
    -y  one output
    -lamba regularization rate
    """
    dw2 = np.zeros(W['W2'].shape)
    dw1 = np.zeros(W['W1'].shape)
    
    y_mat = np.zeros((10,))
    y_mat[y] = 1
    dyhat = -(y_mat-OP)/OP*(1-OP)
    dyhatda1 = ((np.sum(np.exp(A['A2'])) - np.exp(A['A2']))*np.exp(A['A2'])/(np.sum(np.exp(A['A2'])))**2)
    da1w2 = A['A1']
    da1da2 = W['W2']
    da2dw2 = (A['A1']*(1 - A['A1']))
    dw2 = np.outer(dyhat * dyhatda1, da1w2)
    dw1 = np.outer((((np.dot(da1da2.T, dyhat * dyhatda1))) * da2dw2).reshape(W['W2'].shape[1],1), x) 
    return dw2, dw1

def update_mini_batch(x, y, lr, W, lamda):
    dw2 = np.zeros(W['W2'].shape)
    dw1 = np.zeros(W['W1'].shape)
    los = 0.
    for i in range(batch_size):
        A, OP = forward_pass(W,x[i, :])
        deltaw2, deltaw1 = Backword(OP, A, W, x[i, :], y[i])
        dw2 += deltaw2
        dw1 += deltaw1
        los += Loss(OP,y[i])
    
    W['W2'] -= lr*(dw2/batch_size) + (lamda/batch_size)*W['W2']
    W['W1'] -= lr*(dw1/batch_size) + (lamda/batch_size)*W['W1']
    loss = los/batch_size
    return loss
    

def Sigmoid(x):
    sm = 1/(1 + np.exp(-x))
    return sm

def softmax(m):
    out = np.exp(m)/np.sum(np.exp(m), axis = 0)
    return out

def precision(x, y):
    x = np.subtract(x, np.mean(train_x), casting='unsafe') 
    _, OP = forward_pass(W,x)
    Yte_predict = np.argmax(OP, axis = 0)
    return np.mean(Yte_predict == y)

def whole_batch_update(inx, iny, lr, W, lamda, test_x, test_y):
    for j in range(39):
        x = inx[j*batch_size:(j+1)*batch_size, :]
        y = iny[j*batch_size:(j+1)*batch_size]
        loss = update_mini_batch(x, y, lr, W, lamda)
        print('iter %d, loss = %f' %(j, loss))
    print('test precision: %f' %(precision(test_x, test_y)))




train_x, train_y, validation_x, validation_y, test_x, test_y = inportdata(data_dir)
train_x = np.subtract(train_x, np.mean(train_x), casting='unsafe') ##Normlize data

                     

"""
Training process: update weights.
"""
for n in range(2):
    whole_batch_update(train_x, train_y, lr, W, lamda, test_x, test_y)



























for j in range(39):
    x = train_x[j*batch_size:(j+1)*batch_size, :]
    y = train_y[j*batch_size:(j+1)*batch_size]
    loss = update_mini_batch(x, y, lr, W, lamda)
    print('iter %d, loss = %f' %(j, loss))
    precision(test_x, test_y)




x = np.subtract(test_x, np.mean(train_x), casting='unsafe') 

_, OP = forward_pass(W,x)
Yte_predict = np.argmax(OP, axis = 0)
np.mean(Yte_predict == test_y)


_, OP = forward_pass(W,np.subtract(feature.astype(np.float64), np.mean(feature.astype(np.float64)), casting='unsafe'))
Yte_predict = np.argmax(OP, axis = 0)
np.mean(Yte_predict == label)


#########Batch Update#################
feature = feature[:batch_size, :]
feature.astype(np.float64)
feature = np.subtract(feature, np.mean(feature), casting='unsafe') / np.std(feature) ##Normlize data

lr = 0.05

for j in range(500):
    x = feature[j*batch_size:(j+1)*batch_size, :]
    y = label[j*batch_size:(j+1)*batch_size]
    loss = update_mini_batch(lr, W)
    print('iter %d, loss = %f' %(j, loss))

