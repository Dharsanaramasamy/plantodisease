# -*- coding: utf-8 -*-
"""codeathon.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1R3GYzljNX9wpmxjMj3dsL3ZpTJQ5a1ji
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

from google.colab import drive
drive.mount('/content/drive')

directory_doentes = "/content/drive/MyDrive/aiii/ai/Train/Train/Powdery"
directory_saudaveis = "/content/drive/MyDrive/aiii/ai/Train/Train/Healthy"

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

train_datagen=ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,#tilts the image
    zoom_range=0.2,horizontal_flip=True
    )

training_set=train_datagen.flow_from_directory(
    '/content/drive/MyDrive/aiii/ai/Train/Train',
    target_size=(64,64),
    batch_size=32,
    class_mode='binary')

training_set.class_indices

test_datagen=ImageDataGenerator(
rescale=1./255)

test_set=test_datagen.flow_from_directory(
    '/content/drive/MyDrive/aiii/ai/Test/Test',
    target_size=(64,64),
    batch_size=32,
    class_mode='binary')

test_set.class_indices

from tensorflow import keras
cnn=keras.Sequential()
#1 layer
cnn.add(keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu',input_shape=[64,64,3]))
cnn.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=2))
#2 layer
cnn.add(keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu'))
cnn.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=2))
#3 layer
cnn.add(keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu',input_shape=[64,64,3]))
cnn.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=2))

cnn.add(keras.layers.Flatten())
cnn.add(keras.layers.Dense(256,activation='relu'))
cnn.add(keras.layers.Dropout(0.2))
cnn.add(keras.layers.Dense(128,activation='relu'))
cnn.add(keras.layers.Dropout(0.2))
cnn.add(keras.layers.Dense(256,activation='relu'))
cnn.add(keras.layers.Dropout(0.2))
cnn.add(keras.layers.Dense(128,activation='relu'))
cnn.add(keras.layers.Dense(3,activation='sigmoid'))

cnn.summary()

cnn.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics='accuracy')

model_history=cnn.fit(x=training_set,validation_data=test_set,epochs=50)

import pandas as pd
pd.DataFrame(model_history.history).plot()

from keras.utils import load_img,img_to_array
test=load_img('/content/drive/MyDrive/aiii/ai/Train/Train/Rust/82a51a4b035f35fe.jpg',target_size=(64,64))
test

import numpy as np
test=img_to_array(test)
test=np.expand_dims(test,axis=0)
test.shape

result=cnn.predict(test)

result

cnn.save('/content/drive/MyDrive/aiii/Plant disease/cnn.h5')


