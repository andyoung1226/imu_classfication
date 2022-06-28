import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Conv2D, AveragePooling2D, MaxPooling2D
import os
from scipy import io
import numpy as np
import pywt
import rospy
from sensor_msgs.msg import Imu
from std_msgs.msg import Int16

class imu_classification():
    def __init__(self):
        self.imu_sub = rospy.Subscriber('/imu', Imu, self.imu_callback, queue_size=1)
        self.classifi_pub = rospy.Publisher('/cwt_cnn_classfi', Int16, queue_size=1)
        self.imu_data = np.empty(shape=(0, 6))
        self.cwt_data = np.empty(shape=(50, 32, 6, 1))
        
    def reset_imudata(self):
        self.imu_data = np.empty(shape=(0, 6))

    def create_cwt_images(self, X, n_scales, n_samples, wavelet_name="morl"):
        # n_samples = X.shape[0]
        n_signals = X.shape[2]
        n_times = X.shape[1]

        scales = np.arange(1, n_scales + 1)
        X_cwt = np.ndarray(shape=(n_samples, n_scales, n_times, n_signals), dtype='float32')

        for sample in range(n_samples):
            for signal in range(n_signals):
                serie = X[sample, :, signal]
                coeffs, freqs = pywt.cwt(serie, scales, wavelet_name)
                X_cwt[sample, :, :, signal] = coeffs
        
        self.cwt_data = X_cwt

    def imu_callback(self, msg):
        ang_vel = msg.angular_velocity
        lin_acc = msg.linear_acceleration
        a_v_x = ang_vel.x
        a_v_y = ang_vel.y
        a_v_z = ang_vel.z
        l_a_x = lin_acc.x
        l_a_y = lin_acc.y
        l_a_z = lin_acc.z

        self.imu_data = np.append(self.imu_data, np.array([[a_v_x, a_v_y, a_v_z, l_a_x, l_a_y, l_a_z]]), axis=0)
        if len(self.imu_data) == 50:
            #print(self.imu_data.shape, "imu shape")
            
            to_cwt_data = self.imu_data.reshape(50, 6, 1)
            self.create_cwt_images(to_cwt_data, 32, 1)
            predict_data = self.cwt_data
            prediction = cnn_model.predict(predict_data)
            predict_index = int(np.argmax(prediction[0]))
            self.classifi_pub.publish(predict_index)
            if predict_index == 0:
                print("normal driving")
                self.imu_data = np.delete(self.imu_data, (0), axis=0)
            else:
                print("vibaration detected")
                self.reset_imudata()
            #self.imu_data = np.delete(self.imu_data, (0), axis=0)

if __name__ == "__main__":
    cnn_model = models.load_model('/home/wheelchair/catkin_ws/src/imu_classfication/model/cwt_cnn_model')
    os.chdir("/home/wheelchair/catkin_ws/src/imu_classfication/model")
    rospy.init_node("imu_cwt_cnn")
    cnn = imu_classification()
    rospy.spin()