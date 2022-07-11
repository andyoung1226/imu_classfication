#import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
#from tensorflow.keras import models
#from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Conv2D, AveragePooling2D, MaxPooling2D
import os
from scipy import io
import numpy as np
import pywt
import rospy
from sensor_msgs.msg import Imu
from std_msgs.msg import Int16, Float32MultiArray, Float32

class imu_classification():
    def __init__(self):
        self.imu_sub = rospy.Subscriber('/imu', Imu, self.imu_callback, queue_size=1)
        self.data_pub_x = rospy.Publisher('/data_x', Float32MultiArray, queue_size=1)
        self.data_pub_y = rospy.Publisher('/data_y', Float32MultiArray, queue_size=1)
        self.data_pub_z = rospy.Publisher('/data_z', Float32MultiArray, queue_size=1)

        self.imu_data_x = []
        self.imu_data_y = []
        self.imu_data_z = []

        #self.cwt_data = np.empty(shape=(1, 32, 50, 6))
        self.arraydata_x = Float32MultiArray()
        self.arraydata_y = Float32MultiArray()
        self.arraydata_z = Float32MultiArray()

    def reset_imudata(self):
        self.imu_data = np.empty(shape=(0, 6))


    def imu_callback(self, msg):
        ang_vel = msg.angular_velocity
        lin_acc = msg.linear_acceleration
        a_v_x = ang_vel.x
        a_v_y = ang_vel.y
        a_v_z = ang_vel.z
        l_a_x = lin_acc.x
        l_a_y = lin_acc.y
        l_a_z = lin_acc.z
    
        #self.imu_data.append(a_v_x)
        #self.imu_data.append(a_v_y)
        #self.imu_data.append(a_v_z)

        self.imu_data_x.append(l_a_x)
        self.imu_data_y.append(l_a_x)
        self.imu_data_z.append(l_a_x)

        if len(self.imu_data_x) == 100:
            #to_cwt_data = self.imu_data.reshape(1, 50, 6)
            #self.create_cwt_images(to_cwt_data, 32, 1)
            #self.arraydata.data = self.cwt_data.tolist()
            self.arraydata_x.data = self.imu_data_x
            self.arraydata_y.data = self.imu_data_y
            self.arraydata_z.data = self.imu_data_z

            self.data_pub_x.publish(self.arraydata_x)
            self.data_pub_y.publish(self.arraydata_y)
            self.data_pub_z.publish(self.arraydata_z)

            del self.imu_data_x[0]
            del self.imu_data_y[0]
            del self.imu_data_z[0]


if __name__ == "__main__":
    rospy.init_node("data_pub")
    cnn = imu_classification()
    rospy.spin()