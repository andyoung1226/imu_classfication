#!/usr/bin/env python
import rospy
import scipy.io
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Quaternion, Vector3
from std_msgs.msg import Float64
import numpy as np

data_x = np.empty((0,1), float)
data_y = np.empty((0,1), float)
data_z = np.empty((0,1), float)


normal = np.zeros((10000,1), dtype=float)
normal_dic = {"imu_normal_index": normal}
scipy.io.savemat('normal_index.mat', normal_dic)

fault = np.ones((10000,1), dtype=float)
fault_dic = {"imu_normal_index": fault}
scipy.io.savemat('fault_index.mat', fault_dic)

def imu_callback(msg):
	global data
        #angular_velocity = msg.angular_velocity
        linear_acceleration = msg.linear_acceleration
        #a_v_x = angular_velocity.x
        #a_v_y = angular_velocity.y
        #a_v_z = angular_velocity.z
        l_a_x = linear_acceleration.x
        l_a_y = linear_acceleration.y
        l_a_z = linear_acceleration.z

        data_x = np.append(data_x, np.array([l_a_x]))
        data_y = np.append(data_y, np.array([l_a_y]))
        data_z = np.append(data_z, np.array([l_a_z]))

        if len(data_x) == 50000:
                data_x = data_x.reshape(500, 10, 10, 1)
                data_y = data_y.reshape(500, 10, 10, 1)
                data_z = data_z.reshape(500, 10, 10, 1)
                data = np.concatenate((data_x, data_y, data_z), axis=3)
                print(data.shape)
                data_dic = {"imu_normal": data}
                scipy.io.savemat('imu_normal.mat', data_dic)
                print("data_saved")

def listener():
        rospy.init_node('imudata_mat', anonymous=True)
        rospy.Subscriber("/imu", Imu, imu_callback)
        rospy.spin()

if __name__=='__main__':
        print("start")
        listener()

