#!/usr/bin/env python
import rospy
import scipy.io
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Quaternion, Vector3
from std_msgs.msg import Float64
import numpy as np
import keyboard

data_x = np.empty((0,1), float)
data_y = np.empty((0,1), float)
data_z = np.empty((0,1), float)
a = 1

normal = np.zeros((10000,1), dtype=float)
normal_dic = {"imu_normal_index": normal}
scipy.io.savemat('normal_index.mat', normal_dic)

fault = np.ones((10000,1), dtype=float)
fault_dic = {"imu_normal_index": fault}
scipy.io.savemat('fault_index.mat', fault_dic)

def imu_callback(msg):
        global data_x, data_y, data_z, a, data_col
        #angular_velocity = msg.angular_velocity
        linear_acceleration = msg.linear_acceleration
        #a_v_x = angular_velocity.x
        #a_v_y = angular_velocity.y
        #a_v_z = angular_velocity.z
        l_a_x = linear_acceleration.x
        l_a_y = linear_acceleration.y
        l_a_z = linear_acceleration.z

        if data_col == True:
                data_x = np.append(data_x, np.array([l_a_x]))
                data_y = np.append(data_y, np.array([l_a_y]))
                data_z = np.append(data_z, np.array([l_a_z]))

        if len(data_x) == 100:
                data_x = data_x.reshape(1, 10, 10, 1)
                data_y = data_y.reshape(1, 10, 10, 1)
                data_z = data_z.reshape(1, 10, 10, 1)
                data = np.concatenate((data_x, data_y, data_z), axis=3)
                print(data.shape)
                data_dic = {"imu_fault": data}
                scipy.io.savemat('./fault_data/imu_fault_{0}.mat'.format(a), data_dic)
                a += 1
                print("data_saved", a)
                data_x = np.empty((0,1), float)
                data_y = np.empty((0,1), float)
                data_z = np.empty((0,1), float)

def keyCB(msg):
        global data_col
        if msg.data == 1:
                data_col = True
        else:
                data_col = False
def listener():
        rospy.init_node('imudata_mat', anonymous=True)
        rospy.Subscriber("/imu", Imu, imu_callback)
        rospy.spin()

if __name__=='__main__':
        print("start")
        listener()

