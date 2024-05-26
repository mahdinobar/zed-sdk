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

class Server:
    def __init__(self):
        self.img_gray = None
    def img_gray_callback(self, msg):
        # "Store" message received.
        # rospy.loginfo(rospy.get_caller_id() + '+++++++++++I heard %s', data.data)
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        print("timestamp={} [ns]".format(msg.header.stamp.nsecs))
        # print("cv_image=\n",cv_image)
        print("np.asarray(cv_image)=\n", np.asarray(cv_image))


if __name__ == '__main__':
    rospy.init_node('listener')
    server = Server()
    rospy.Subscriber('/zedxm/zed_node/rgb/image_rect_gray', Image, server.img_gray_callback)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()