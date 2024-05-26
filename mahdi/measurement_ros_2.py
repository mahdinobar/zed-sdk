########################################################################
#
# Copyright (c) 2024, Mahdi Nobar.
#
# All rights reserved.
#
########################################################################

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np

def callback(data):
    # rospy.loginfo(rospy.get_caller_id() + '+++++++++++I heard %s', data.data)
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
    print("cv_image=\n",cv_image)
    print("np.asarray(cv_image)=\n",np.asarray(cv_image))

def listener():

    rospy.init_node('listener')

    rospy.Subscriber('/zedxm/zed_node/rgb/image_rect_gray', Image, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
