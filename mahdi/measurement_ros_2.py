########################################################################
#
# Copyright (c) 2024, Mahdi Nobar.
#
# All rights reserved.
#
########################################################################

import rospy
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import numpy as np
import ros_numpy

class Server:
    def __init__(self):
        self.img_gray = None
    def img_gray_callback(self, msg):
        # "Store" message received.
        # rospy.loginfo(rospy.get_caller_id() + '+++++++++++I heard %s', data.data)
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        print("timestamp_gray_image={} [ns]".format(msg.header.stamp.nsecs))
        # print("cv_image=\n",cv_image)
        print("np.asarray(gray-image)=\n", np.asarray(cv_image))
    def img_color_callback(self, msg):
        # "Store" message received.
        # rospy.loginfo(rospy.get_caller_id() + '+++++++++++I heard %s', data.data)
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        print("timestamp_color_image={} [ns]".format(msg.header.stamp.nsecs))
        # print("cv_image=\n",cv_image)
        print("color_image=\n", np.asarray(cv_image))
    def img_pointcloud_callback(self, msg):
        # "Store" message received.
        # rospy.loginfo(rospy.get_caller_id() + '+++++++++++I heard %s', data.data)
        point_cloud=ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg)
        print("timestamp_pointcloud={} [ns]".format(msg.header.stamp.nsecs))
        # print("cv_image=\n",cv_image)
        print("pointcloud=\n", point_cloud)

        x=10
        y=200
        arrayPosition = y * msg.row_step + x * msg.point_step
        arrayPosX = arrayPosition + msg.fields[0].offset
        arrayPosY = arrayPosition + msg.fields[1].offset
        arrayPosZ = arrayPosition + msg.fields[2].offset
        X=msg.data[arrayPosX]
        Y=msg.data[arrayPosY]
        Z=msg.data[arrayPosZ]


if __name__ == '__main__':
    rospy.init_node('listener')
    server = Server()
    rospy.Subscriber('/zedxm/zed_node/rgb/image_rect_gray', Image, server.img_gray_callback)
    rospy.Subscriber('/zedxm/zed_node/rgb/image_rect_color', Image, server.img_color_callback)
    rospy.Subscriber('/zedxm/zed_node/point_cloud/cloud_registered', PointCloud2, server.img_pointcloud_callback)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()