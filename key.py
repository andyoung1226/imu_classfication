import rospy
from std_msgs.msg import Int8

pub = rospy.Publisher('/keyboard', Int8, queue_size=10)
rospy.init_node('keypub')
r = rospy.Rate(10) # 10hz
while not rospy.is_shutdown():
    a = input()
    if a == 'a':
        data = 1
    else:
        data = 0
    pub.publish(data)
    r.sleep()