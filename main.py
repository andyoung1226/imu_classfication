# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Conv2D, AveragePooling2D, MaxPooling2D
import os
from scipy import io
import numpy as np

print(tf.__version__)
imu_data = io.loadmat('C:/Users/UNICON/Desktop/DY/Works/wheelchair_tp/wheelchair_imu_classification/imu_normal.mat')
imu_index = io.loadmat('C:/Users/UNICON/Desktop/DY/Works/wheelchair_tp/wheelchair_imu_classification/normal_index.mat')
fault_data = io.loadmat('C:/Users/UNICON/Desktop/DY/Works/wheelchair_tp/wheelchair_imu_classification/imu_fault.mat')
fault_index = io.loadmat('C:/Users/UNICON/Desktop/DY/Works/wheelchair_tp/wheelchair_imu_classification/fault_index.mat')

imu_data = imu_data['imu_normal']
imu_index = imu_index['imu_normal_index']
fault_data = fault_data['imu_fault']
fault_index = fault_index['imu_normal_index']

a = 0
Xtrain_normal = np.zeros((8951, 50, 6))
Xtrain_fault = np.zeros((8951, 50, 6))
Xtest_normal = np.zeros((951, 50, 6))
Xtest_fault = np.zeros((951, 50, 6))

for i in range(8951):
    Xtrain_normal[i, :, :] = imu_data[a:a + 50, :]
    Xtrain_fault[i, :, :] = fault_data[a:a + 50, :]
    a += 1
a = 9000
for i in range(951):
    Xtest_normal[i, :, :] = imu_data[a:a + 50, :]
    Xtest_fault[i, :, :] = fault_data[a:a + 50, :]
    a += 1

train_data = np.concatenate((Xtrain_normal, Xtrain_fault), axis=0)
train_index = np.concatenate((imu_index[0:8951, :], fault_index[0:8951, :]), axis=0)
test_data = np.concatenate((Xtest_normal, Xtest_fault), axis=0)
test_index = np.concatenate((imu_index[9000:9951, :], fault_index[9000:9951, :]), axis=0)

Xtrain = train_data.reshape(-1, 50, 6, 1)
Xtest = test_data.reshape(-1, 50, 6, 1)
Ytrain = train_index.reshape(-1, 1)
Ytest = test_index.reshape(-1, 1)

train_num = Ytrain.shape
test_num = Ytest.shape
print(Xtrain.shape, Ytrain.shape)

model = models.Sequential([
    Conv2D(16, (3, 2), input_shape=(50, 6, 1),activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(32, (3, 2), activation='relu', padding='same'),
    MaxPooling2D((2, 2), padding='same'),
    #BatchNormalization(),
    Conv2D(64, (3, 2), activation='relu', padding='same'),

    #BatchNormalization(),
    #AveragePooling2D((1, 1)),
    Flatten(),
    Dense(256, activation='relu'),
    #BatchNormalization(),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(Xtrain, Ytrain, epochs=12, batch_size=200, validation_data=(Xtest, Ytest), shuffle=True)

# tf.keras.utils.plot_model(model, to_file='C:/Users/UNICON/Desktop/model.png', show_shapes=True)
#print(model.summary())

model.save('model/cnn_model')
loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs_range = range(12)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Traning Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Traing and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training_loss')
plt.plot(epochs_range, val_loss, label='Validation_loss')
plt.legend(loc='upper right')
plt.title('Training and Validation loss')
plt.show()
