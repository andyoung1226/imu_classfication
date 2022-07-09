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
        self.imu_sub = rospy.Subscriber('/data', Float32MultiArray, self.data_callback, queue_size=1)
        self.classifi_pub = rospy.Publisher('/cwt_cnn_classfi', Int16, queue_size=1)
        self.sos = signal.butter(11, 15, 'hp', fs=50, output='sos')
        self.filtered_data = np.empty(shape=(1, 50, 6))

    def data_callback(self, msg):
        to_cwt_data = np.array(msg.data)
        to_cwt_data = to_cwt_data.reshape(1, 50, 6)
        for i in range(6):
            self.filtered_data[:, :, i] = signal.sosfilt(self.sos, to_cwt_data[:, :, i])
        predict_data = self.filtered_data
        predict_data = predict_data.reshape(1, 50, 6, 1)
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