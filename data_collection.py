#!/usr/bin/env python
import rospy
import scipy.io
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Quaternion, Vector3
from std_msgs.msg import Float64
import numpy as np

data = np.empty((0,6), float)

normal = np.zeros((10000,1), dtype=float)
normal_dic = {"imu_normal_index": normal}
scipy.io.savemat('normal_index.mat', normal_dic)

fault = np.ones((10000,1), dtype=float)
fault_dic = {"imu_normal_index": fault}
scipy.io.savemat('fault_index.mat', fault_dic)

def imu_callback(msg):
	global data

        angular_velocity = msg.angular_velocity
        linear_acceleration = msg.linear_acceleration

        a_v_x = angular_velocity.x
        a_v_y = angular_velocity.y
        a_v_z = angular_velocity.z
        l_a_x = linear_acceleration.x
        l_a_y = linear_acceleration.y
        l_a_z = linear_acceleration.z

        data = np.append(data, np.array([[a_v_x, a_v_y, a_v_z, l_a_x, l_a_y, l_a_z]]), axis=0)

        if len(data) == 10000:
                data_reshape = data.reshape(-1, 6)
                print(data_reshape.shape)
                print(data_reshape)
                data_dic = {"imu_normal": data_reshape}
                scipy.io.savemat('imu_normal.mat', data_dic)
                print("data_saved")

def listener():
        rospy.init_node('imudata_mat', anonymous=True)
        rospy.Subscriber("/imu", Imu, imu_callback)
        rospy.spin()

if __name__=='__main__':
        print("start")
        listener()

