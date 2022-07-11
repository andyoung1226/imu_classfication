import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Conv2D, AveragePooling2D, MaxPooling2D
import os
from scipy import io, signal
import numpy as np
import pywt
import rospy
from sensor_msgs.msg import Imu
from std_msgs.msg import Int16, Float32MultiArray

class imu_classification():
    def __init__(self):
        self.imu_sub_x = rospy.Subscriber('/data_x', Float32MultiArray, self.data_callback_x, queue_size=1)
        self.imu_sub_y = rospy.Subscriber('/data_y', Float32MultiArray, self.data_callback_y, queue_size=1)
        self.imu_sub_z = rospy.Subscriber('/data_z', Float32MultiArray, self.data_callback_z, queue_size=1)
        self.classifi_pub = rospy.Publisher('/cwt_cnn_classfi', Int16, queue_size=1)
        self.sos = signal.butter(11, 15, 'hp', fs=50, output='sos')
        self.filtered_data = np.empty(shape=(1, 50, 6))
        self.data_x = np.empty(shape=(1, 10, 10, 1))
        self.data_y = np.empty(shape=(1, 10, 10, 1))
        self.data_z = np.empty(shape=(1, 10, 10, 1))
        self.data = np.empty(shape=(1, 10, 10, 3))
    def data_callback_x(self, msg):
        self.data_x = msg.data

    def data_callback_y(self, msg):
        self.data_y = msg.data

    def data_callback_z(self, msg):
        self.data_z = msg.data
        self.data = np.concatenate((self.data_x, self.data_y, self.data_z), axis=3)
        #to_cwt_data = np.array(msg.data)
        #to_cwt_data = to_cwt_data.reshape(1, 50, 6)
        #for i in range(6):
        #    self.filtered_data[:, :, i] = signal.sosfilt(self.sos, to_cwt_data[:, :, i])
        #predict_data = self.filtered_data
        #predict_data = predict_data.reshape(1, 50, 6, 1)
        predict_data = self.data
        prediction = cnn_model.predict(predict_data)
        predict_index = int(np.argmax(prediction[0]))
        self.classifi_pub.publish(predict_index)
        if predict_index == 0:
            print("normal driving")
        else:
            print("vibaration detected")
        #self.imu_data = np.delete(self.imu_data, (0), axis=0)


if __name__ == "__main__":
    cnn_model = models.load_model('/home/wheelchair/catkin_ws/src/imu_classfication/model/cnn_model')
    os.chdir("/home/wheelchair/catkin_ws/src/imu_classfication/model")
    rospy.init_node("imu_cwt_cnn")
    cnn = imu_classification()
    rospy.spin()