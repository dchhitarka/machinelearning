#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 23:06:11 2019

@author: dchhitarka
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras.optimizers import adam
from keras.losses import categorical_crossentropy
import numpy as np


data = mnist.load_data()
(x_train, y_train), (x_test, y_test) = data

n_classes = len(np.unique(y_train))

x_train = x_train.reshape(len(x_train),-1)
x_test = x_test.reshape(len(x_test),-1)

#Normalize
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
x_train = scale.fit_transform(x_train)
x_test = scale.fit_transform(x_test)

y_train = to_categorical(y_train, num_classes=n_classes)
y_test = to_categorical(y_test, num_classes=n_classes)

model = Sequential()
model.add(Dense(units=10, kernel_initializer='uniform', activation='relu', input_dim=784))
model.add(Dense(units=10, kernel_initializer='uniform', activation='relu'))
model.add(Dense(units=10, kernel_initializer='uniform', activation='softmax'))

model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])

model.summary()

model.fit(x_train, y_train, batch_size=128, epochs=20, validation_split=0.2)

score, acc = model.evaluate(x_test, y_test, batch_size=128)
print('Test score:', score)
print('Test accuracy:', acc)
"""
Train on 48000 samples, validate on 12000 samples
Epoch 1/20
48000/48000 [==============================] - 7s 150us/step - loss: 1.1522 - acc: 0.6241 - val_loss: 0.4344 - val_acc: 0.8709
Epoch 2/20
48000/48000 [==============================] - 2s 49us/step - loss: 0.3804 - acc: 0.8867 - val_loss: 0.3153 - val_acc: 0.9044
Epoch 3/20
48000/48000 [==============================] - 3s 71us/step - loss: 0.3111 - acc: 0.9062 - val_loss: 0.2882 - val_acc: 0.9133
Epoch 4/20
48000/48000 [==============================] - 2s 45us/step - loss: 0.2816 - acc: 0.9152 - val_loss: 0.2690 - val_acc: 0.9189
Epoch 5/20
48000/48000 [==============================] - 2s 36us/step - loss: 0.2610 - acc: 0.9224 - val_loss: 0.2586 - val_acc: 0.9221
Epoch 6/20
48000/48000 [==============================] - 2s 37us/step - loss: 0.2459 - acc: 0.9274 - val_loss: 0.2548 - val_acc: 0.9226
Epoch 7/20
48000/48000 [==============================] - 2s 37us/step - loss: 0.2339 - acc: 0.9303 - val_loss: 0.2409 - val_acc: 0.9295
Epoch 8/20
48000/48000 [==============================] - 2s 37us/step - loss: 0.2239 - acc: 0.9347 - val_loss: 0.2415 - val_acc: 0.9297
Epoch 9/20
48000/48000 [==============================] - 2s 38us/step - loss: 0.2157 - acc: 0.9360 - val_loss: 0.2355 - val_acc: 0.9324
Epoch 10/20
48000/48000 [==============================] - 2s 37us/step - loss: 0.2096 - acc: 0.9376 - val_loss: 0.2346 - val_acc: 0.9318
Epoch 11/20
48000/48000 [==============================] - 2s 37us/step - loss: 0.2030 - acc: 0.9398 - val_loss: 0.2304 - val_acc: 0.9345
Epoch 12/20
48000/48000 [==============================] - 2s 37us/step - loss: 0.1976 - acc: 0.9419 - val_loss: 0.2311 - val_acc: 0.9352
Epoch 13/20
48000/48000 [==============================] - 2s 37us/step - loss: 0.1933 - acc: 0.9425 - val_loss: 0.2301 - val_acc: 0.9343
Epoch 14/20
48000/48000 [==============================] - 2s 37us/step - loss: 0.1896 - acc: 0.9436 - val_loss: 0.2264 - val_acc: 0.9369
Epoch 15/20
48000/48000 [==============================] - 2s 37us/step - loss: 0.1853 - acc: 0.9447 - val_loss: 0.2266 - val_acc: 0.9355
Epoch 16/20
48000/48000 [==============================] - 2s 37us/step - loss: 0.1816 - acc: 0.9464 - val_loss: 0.2258 - val_acc: 0.9376
Epoch 17/20
48000/48000 [==============================] - 2s 37us/step - loss: 0.1778 - acc: 0.9476 - val_loss: 0.2270 - val_acc: 0.9365
Epoch 18/20
48000/48000 [==============================] - 2s 37us/step - loss: 0.1747 - acc: 0.9479 - val_loss: 0.2257 - val_acc: 0.9372
Epoch 19/20
48000/48000 [==============================] - 2s 37us/step - loss: 0.1719 - acc: 0.9485 - val_loss: 0.2252 - val_acc: 0.9369
Epoch 20/20
48000/48000 [==============================] - 2s 37us/step - loss: 0.1685 - acc: 0.9489 - val_loss: 0.2239 - val_acc: 0.9375
10000/10000 [==============================] - 0s 17us/step
Test score: 0.22224724568128587
Test accuracy: 0.9356
"""

"""
model.fit(x_train, y_train, batch_size=128, epochs=10, validation_split=0.2)

score, acc = model.evaluate(x_test, y_test, batch_size=128)
print('Test score:', score)
print('Test accuracy:', acc)

Train on 48000 samples, validate on 12000 samples
Epoch 1/10
48000/48000 [==============================] - 2s 35us/step - loss: 1.1752 - acc: 0.6336 - val_loss: 0.4801 - val_acc: 0.8593
Epoch 2/10
48000/48000 [==============================] - 1s 28us/step - loss: 0.4023 - acc: 0.8829 - val_loss: 0.3147 - val_acc: 0.9071
Epoch 3/10
48000/48000 [==============================] - 1s 28us/step - loss: 0.3057 - acc: 0.9099 - val_loss: 0.2745 - val_acc: 0.9165
Epoch 4/10
48000/48000 [==============================] - 1s 27us/step - loss: 0.2733 - acc: 0.9184 - val_loss: 0.2597 - val_acc: 0.9214
Epoch 5/10
48000/48000 [==============================] - 1s 28us/step - loss: 0.2542 - acc: 0.9243 - val_loss: 0.2478 - val_acc: 0.9246
Epoch 6/10
48000/48000 [==============================] - 1s 27us/step - loss: 0.2405 - acc: 0.9275 - val_loss: 0.2400 - val_acc: 0.9287
Epoch 7/10
48000/48000 [==============================] - 1s 30us/step - loss: 0.2292 - acc: 0.9312 - val_loss: 0.2354 - val_acc: 0.9295
Epoch 8/10
48000/48000 [==============================] - 2s 48us/step - loss: 0.2202 - acc: 0.9334 - val_loss: 0.2346 - val_acc: 0.9319
Epoch 9/10
48000/48000 [==============================] - 2s 35us/step - loss: 0.2124 - acc: 0.9359 - val_loss: 0.2276 - val_acc: 0.9335
Epoch 10/10
48000/48000 [==============================] - 2s 40us/step - loss: 0.2053 - acc: 0.9378 - val_loss: 0.2290 - val_acc: 0.9335
10000/10000 [==============================] - 0s 16us/step
Test score: 0.23759391102790833
Test accuracy: 0.9283
"""