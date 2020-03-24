#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 20:01:47 2019

@author: dchhitarka
"""
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation
from keras.datasets import mnist #, cifar10 
(x_train, y_train), (x_test, y_test) = mnist.load_data()

class LeNet:
    @staticmethod
    def build(input_shape, classes):
        model = Sequential()
        model.add(Conv2D(20, kernel_size=5, padding="same", input_shape=input_shape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        
        model.add(Conv2D(50, kernel_size=5, border_mode="same"))
        model.add(Activation("same"))
        model.add(MaxPooling2D(pool_size=(2,2), stride=(2,2)))
        
        model.add(Flatten())
        model.add(Dense(500, activation="relu"))
        
        model.add(Dense(classes, activation="softmax"))
        
        return model
    
