#!/usr/bin/env python3

########################################################################
#
# Copyright (c) 2024, Mahdi Nobar.
#
# All rights reserved.
#
########################################################################

import rospy
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float64MultiArray

from cv_bridge import CvBridge
import numpy as np
import ros_numpy
from message_filters import TimeSynchronizer, Subscriber
import message_filters
from dt_apriltags import Detector
import cv2
from std_msgs.msg import String
import sensor_msgs.point_cloud2


class Server:
    def __init__(self):
        self.img_gray = None
        self.bridge = CvBridge()
        self.at_detector = Detector(searchpath=['apriltags'],
                                    families='tag36h11',
                                    nthreads=1,
                                    quad_decimate=1.0,
                                    quad_sigma=0.0,
                                    refine_edges=1,
                                    decode_sharpening=0.25,
                                    debug=0)
        self.T_ca_ftc2 = Float64MultiArray()
        self.p_obj_ca = Vector3()
        # self.cx = 479.4909973144531
        # self.cy = 299.3664855957031
        # self.fx = 367.72650146484375
        # self.fy = 367.6239929199219

        # _b from /zedxm/zed_node/rgb/camera_info
        self.cx = 472.77484130859375
        self.cy = 281.85357666015625
        self.fx = 377.4441223144531
        self.fy = 377.4441223144531

        # # _c from file /usr/local/zed/settings/SN58039767.conf  for [LEFT_CAM_FHD]
        # self.cx = 958.982
        # self.cy = 538.733
        # self.fx = 735.453
        # self.fy = 735.248

        # # _d from /zedxm/zed_node/left_raw/camera_info
        # self.cx = 479.4909973144531
        # self.cy = 269.3664855957031
        # self.fx = 367.72650146484375
        # self.fy = 367.6239929199219

        # # _e from /zedxm/zed_node/left/camera_info
        # self.cx = 472.77484130859375
        # self.cy = 281.85357666015625
        # self.fx = 377.4441223144531
        # self.fy = 377.4441223144531


        # _f from /zedxm/zed_node/depth/camera_info
        self.cx = 487.22515869140625
        self.cy = 258.14642333984375
        self.fx = 377.4441223144531
        self.fy = 377.4441223144531

        self.debug = 0

        # self.pub_p_obj_ca = rospy.Publisher('p_obj_ca', Vector3, queue_size=10)
        # self.pub_T_ca_ftc2 = rospy.Publisher('T_ca_ftc2', Float64MultiArray, queue_size=10)
        # self.p_obj_ca.x = 0
        # self.p_obj_ca.y = 0
        # self.p_obj_ca.z = 0

    def cv2_imshow(self, a, window_name="image"):
        """A replacement for cv2.imshow() for use in Jupyter notebooks.
        Args:
        a : np.ndarray. shape (N, M) or (N, M, 1) is an NxM grayscale image. shape
          (N, M, 3) is an NxM BGR color image. shape (N, M, 4) is an NxM BGRA color
          image.
        """
        # cv2 stores colors as BGR; convert to RGB
        if a.ndim == 3:
            a = a.clip(0, 255).astype('uint8')
            if a.shape[2] == 4:
                a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA)
            else:
                a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
        if a.ndim == 2:
            a = a.clip(100, 500).astype('uint8')
        cv2.imshow(window_name, a)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def gotdata(self, gray_image, color_image, depth_image, point_cloud):
        if self.debug:
            print("timestamp_gray_image={} [ns]".format(gray_image.header.stamp.nsecs))
            print("timestamp_color_image={} [ns]".format(color_image.header.stamp.nsecs))
            print("timestamp_point_cloud={} [ns]".format(point_cloud.header.stamp.nsecs))

        gray_image = self.bridge.imgmsg_to_cv2(gray_image, desired_encoding='passthrough')
        if self.debug:
            print("gray-image=\n", gray_image)

        color_image = self.bridge.imgmsg_to_cv2(color_image, desired_encoding='passthrough')
        if self.debug:
            print("color_image=\n", color_image)
        color_image_copy = color_image.copy()
        cv2.imwrite("/home/user/code/zed-sdk/mahdi/log/image_left_1.jpeg", color_image_copy)
        # depth_image = self.bridge.imgmsg_to_cv2(depth_image, desired_encoding='passthrough')
        depth_image = np.array(self.bridge.imgmsg_to_cv2(depth_image, desired_encoding='passthrough'), dtype=np.float32)
        depth_image_copy = depth_image.copy()

        # x=10
        # y=200
        # arrayPosition = y * point_cloud.row_step + x * point_cloud.point_step
        # arrayPosX = arrayPosition + point_cloud.fields[0].offset
        # arrayPosY = arrayPosition + point_cloud.fields[1].offset
        # arrayPosZ = arrayPosition + point_cloud.fields[2].offset
        # X=point_cloud.data[arrayPosX]
        # Y=point_cloud.data[arrayPosY]
        # Z=point_cloud.data[arrayPosZ]
        # print(X,y,Z)

        tags = self.at_detector.detect(gray_image, False, camera_params=None)
        point_cloud_value = np.zeros((2, 3))
        tag_idx = 0
        for tag in tags:
            for idx in range(len(tag.corners)):
                if self.debug:
                    # print(
                    #     "!!corner detected on image plane location = ({},{}) [pxls].".format(
                    #         tag.corners[idx, 0], tag.corners[idx, 1]))
                    cv2.line(color_image_copy, tuple(tag.corners[idx - 1, :].astype(int)),
                             tuple(tag.corners[idx, :].astype(int)),
                             (0, 255, 0))
                    cv2.drawMarker(color_image_copy, tuple(tag.corners[idx, :].astype(int)), color=(255, 0, 0))
                    cv2.putText(color_image_copy, str(idx),
                                org=(tag.corners[idx, 0].astype(int) + 3, tag.corners[idx, 1].astype(int) + 3),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.5,
                                color=(255, 0, 0))
                    cv2.line(depth_image_copy, tuple(tag.corners[idx - 1, :].astype(int)),
                             tuple(tag.corners[idx, :].astype(int)),
                             (0, 255, 0))
                    cv2.drawMarker(depth_image_copy, tuple(tag.corners[idx, :].astype(int)), color=(255, 0, 0))
                    cv2.putText(depth_image_copy, str(idx),
                                org=(tag.corners[idx, 0].astype(int) + 3, tag.corners[idx, 1].astype(int) + 3),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.5,
                                color=(255, 0, 0))
            if self.debug:
                cv2.putText(color_image_copy, str(tag_idx),
                            org=(tag.corners[0, 0].astype(int) - 5, tag.corners[0, 1].astype(int) - 5),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1.5,
                            color=(200, 0, 200))
            R_cam2gripper= np.array([[-0.02499009, 0.68761139, -0.72564872], [-0.99963839, -0.02439711, 0.01130757], [-0.00992852, 0.7256689, 0.68797244]])
            t_cam2gripper= np.array([[ 0.12082282], [ 0.02026399], [-0.11518659]])
            T_ca_ftc2 = np.vstack((np.hstack((R_cam2gripper, t_cam2gripper.reshape(3, 1))), np.array([0, 0, 0, 1])))
            self.T_ca_ftc2.data = T_ca_ftc2.flatten()

            x_p_obj_img, y_p_obj_img = tag.corners[3]
            # TODO check u and v
            v = x_p_obj_img
            u = y_p_obj_img
            depth_tmp = depth_image_copy[int(u) - 3:int(u) + 3, int(v) - 3:int(v) + 3]
            # TODO 0.400 [m] prior magic number
            acceptable_idx = depth_tmp < 0.400
            Z = np.mean(depth_tmp[acceptable_idx])
            X = Z * (u - self.cx) / self.fx
            Y = Z * (v - self.cy) / self.fy
            p_obj_ca = np.array([X, Y, Z])
            self.p_obj_ca.x = p_obj_ca[0]
            self.p_obj_ca.y = p_obj_ca[1]
            self.p_obj_ca.z = p_obj_ca[2]
            
        if self.debug:
            self.cv2_imshow(color_image_copy, window_name="left image")
            cv2.imwrite("/home/user/code/zed-sdk/mahdi/log/image_left.jpeg", color_image_copy)
            self.cv2_imshow(depth_image_copy, window_name="depth image")
            print("=============================")

        # self.pub_p_obj_ca.publish(self.p_obj_ca)
        # self.pub_T_ca_ftc2.publish(self.T_ca_ftc2)
        # info = "p_obj_ca.x={}".format(self.p_obj_ca.x)
        # rospy.loginfo(info)


if __name__ == '__main__':

    rospy.init_node('my_node')
    server = Server()
    gray_image_listener = message_filters.Subscriber('/zedxm/zed_node/rgb/image_rect_gray', Image)
    color_image_listener = message_filters.Subscriber('/zedxm/zed_node/rgb/image_rect_color', Image)
    depth_image_listener = message_filters.Subscriber('/zedxm/zed_node/depth/depth_registered', Image)
    point_cloud_listener = message_filters.Subscriber('/zedxm/zed_node/point_cloud/cloud_registered', PointCloud2)
    ts = message_filters.ApproximateTimeSynchronizer(
        [gray_image_listener, color_image_listener, depth_image_listener, point_cloud_listener],
        10, 0.1)
    ts.registerCallback(server.gotdata)

    # publish p_obj_ca, T_ca_ftc2
    pub_p_obj_ca = rospy.Publisher('p_obj_ca', Vector3, queue_size=10)
    pub_T_ca_ftc2 = rospy.Publisher('T_ca_ftc2', Float64MultiArray, queue_size=10)
    # pub = rospy.Publisher('chatter222', String, queue_size=10)
    rate = rospy.Rate(1)  # 10hz
    while not rospy.is_shutdown():
        # hello_str = "2222222222222hello world %s" % rospy.get_time()
        # rospy.loginfo(hello_str)
        # pub.publish(hello_str)

        # server.p_obj_ca=Vector3
        # server.p_obj_ca.x = 0
        # server.p_obj_ca.y = 0
        # server.p_obj_ca.z = 0

        pub_p_obj_ca.publish(server.p_obj_ca)
        pub_T_ca_ftc2.publish(server.T_ca_ftc2)
        if 0:
            info = "p_obj_ca.x={}".format(server.p_obj_ca.x)
            rospy.loginfo(info)
            info = "p_obj_ca.y={}".format(server.p_obj_ca.y)
            rospy.loginfo(info)
            info = "p_obj_ca.z={}".format(server.p_obj_ca.z)
            rospy.loginfo(info)
            info = "T_ca_ftc2.data={}".format(server.T_ca_ftc2.data)
            rospy.loginfo(info)
        # try:
        #     info = "T_ca_ftc2.data[1]={}".format(server.T_ca_ftc2.data[0])
        #     rospy.loginfo(info)
        # except:
        #     pass
        rate.sleep()

    rospy.spin()
