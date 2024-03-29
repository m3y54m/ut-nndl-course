# -*- coding: utf-8 -*-
"""miniproj1-4-e.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1WEaQDFz_H0sJWH9uancIXiLvBEefGTWF
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D 

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os

print(tf.VERSION)
print(tf.keras.__version__)

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Reduce size of dataset to 600 by extracting random values
idx_train = np.random.choice(np.arange(x_train.shape[0]), size=500)
idx_test = np.random.choice(np.arange(x_test.shape[0]), size=100)
x_train = np.array([x_train[i] for i in idx_train])
y_train = np.array([y_train[i] for i in idx_train])
x_test = np.array([x_test[i] for i in idx_test])
y_test = np.array([y_test[i] for i in idx_test])

print('x_train shape:', x_train.shape)

trainset_size = x_train.shape[0]
testset_size = x_test.shape[0]
batch_size = 100
steps_per_epoch = np.ceil(trainset_size / batch_size).astype('int')
num_classes = 10
epochs = 100
# input shape: (32, 32, 3)
input_shape = x_train.shape[1:]

data_augmentation = False
#num_predictions = 20

print(trainset_size, 'train samples')
print(testset_size, 'test samples')

# Convert class vectors to binary (one-hot) class matrices.
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

tf.keras.backend.clear_session()

model = Sequential()

model.add(Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

# initiate RMSprop optimizer
#opt = tf.keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)

# Let's train the model using AdamOptimizer
model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Normalize input numbers
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

results = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

# Accuracy for Training Dataset
plt_acc = results.history["acc"]
# Accuracy for Validation (Test) Dataset
plt_val_acc = results.history["val_acc"]

num_epochs = len(plt_val_acc)
plt_epoch = np.arange(1, num_epochs+1, 1, dtype=int)

plt.figure(1, figsize=(10, 6))
plt.plot(plt_epoch, plt_acc, color='c', linewidth=4, label='Training Dataset')
plt.plot(plt_epoch, plt_val_acc, color='m', linewidth=4, label='Test Dataset')
plt.xlim(1, epochs)
plt.xticks(np.arange(0, num_epochs+1, 10, dtype=int))
plt.legend(loc='best', fontsize=15)
plt.xlabel(r'$Epochs$', fontsize=15)
plt.ylabel(r'$Accuracy$', fontsize=15)
plt.grid(True)
plt.show()