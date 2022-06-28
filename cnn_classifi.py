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
        self.classifi_pub = rospy.Publisher('/cnn_classfi', Int16, queue_size=1)
        self.imu_data = np.empty(shape=(6,50))
        self.cnn_model = models.load_model('/model/cnn_model')

    def imu_callback(self, msg):
        ang_vel = msg.angular_velocty
        lin_acc = msg.linear_acceleration
        a_v_x = ang_vel.x
        a_v_y = ang_vel.y
        a_v_z = ang_vel.z
        l_a_x = lin_acc.x
        l_a_y = lin_acc.y
        l_a_z = lin_acc.z

        self.imu_data = np.append(self.imu_data, np.array([a_v_x, a_v_y, a_v_z, l_a_x, l_a_y, l_a_z]), axis=0)
        if len(self.imu_data) == 50:
            prediction = self.cnn_model.predict(self.imu_data)
            print(prediction)
            #self.classifi_pub.publsih(prediction)
            self.imu_data = np.delete(self.imu_data, (0), axis=0)

if __name__ == "__main__":
    rospy.init_node("imu_cnn")
    cnn = imu_classification()
    rospy.spin()