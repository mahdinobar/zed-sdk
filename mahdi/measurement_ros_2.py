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
from message_filters import TimeSynchronizer, Subscriber
import message_filters

class Server:
    def __init__(self):
        self.img_gray = None
        self.bridge = CvBridge()

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
        # pixel indeces
        x=10
        y=200
        arrayPosition = y * msg.row_step + x * msg.point_step
        arrayPosX = arrayPosition + msg.fields[0].offset
        arrayPosY = arrayPosition + msg.fields[1].offset
        arrayPosZ = arrayPosition + msg.fields[2].offset
        # XYZ 3D poit values
        X=msg.data[arrayPosX]
        Y=msg.data[arrayPosY]
        Z=msg.data[arrayPosZ]

    def gotdata(self, msg1, msg2, msg3):
        print("timestamp_msg1={} [ns]".format(msg1.header.stamp.nsecs))
        print("timestamp_msg2={} [ns]".format(msg2.header.stamp.nsecs))
        print("timestamp_msg3={} [ns]".format(msg3.header.stamp.nsecs))

        gray_image = self.bridge.imgmsg_to_cv2(msg1, desired_encoding='passthrough')
        print("gray-image=\n", np.asarray(gray_image))


        color_image = self.bridge.imgmsg_to_cv2(msg2, desired_encoding='passthrough')
        print("color_image=\n", np.asarray(color_image))

        x=10
        y=200
        arrayPosition = y * msg3.row_step + x * msg3.point_step
        arrayPosX = arrayPosition + msg3.fields[0].offset
        arrayPosY = arrayPosition + msg3.fields[1].offset
        arrayPosZ = arrayPosition + msg3.fields[2].offset
        X=msg3.data[arrayPosX]
        Y=msg3.data[arrayPosY]
        Z=msg3.data[arrayPosZ]
        print(X,y,Z)

if __name__ == '__main__':
    rospy.init_node('listener')
    server = Server()
    # rospy.Subscriber('/zedxm/zed_node/rgb/image_rect_gray', Image, server.img_gray_callback)
    # rospy.Subscriber('/zedxm/zed_node/rgb/image_rect_color', Image, server.img_color_callback)
    # rospy.Subscriber('/zedxm/zed_node/point_cloud/cloud_registered', PointCloud2, server.img_pointcloud_callback)

    # tss = TimeSynchronizer(Subscriber('/zedxm/zed_node/rgb/image_rect_gray', Image),
    #                        Subscriber('/zedxm/zed_node/rgb/image_rect_color', Image),
    #                        Subscriber('/zedxm/zed_node/point_cloud/cloud_registered', PointCloud2))
    # tss.registerCallback(server.gotdata)


    sub1 = message_filters.Subscriber('/zedxm/zed_node/rgb/image_rect_gray', Image)
    sub2 = message_filters.Subscriber('/zedxm/zed_node/rgb/image_rect_color', Image)
    sub3 = message_filters.Subscriber('/zedxm/zed_node/point_cloud/cloud_registered', PointCloud2)


    ts = message_filters.TimeSynchronizer([sub1, sub2, sub3], 10)

    ts.registerCallback(server.gotdata)

    rospy.spin()
    
