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
        self.T_ftc_ca = Float64MultiArray()
        self.p_obj_ca = Vector3()
        # _f from /zedxm/zed_node/depth/camera_info
        self.cx = 487.22515869140625
        self.cy = 258.14642333984375
        self.fx = 377.4441223144531
        self.fy = 377.4441223144531
        self.debug = 0

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
        N = 201
        log_data = 1
        O_T_EE = np.array([0.058487896153253345, -0.2934875842206725, 0.9541619101667643, 0.0, -0.7479825753354957,
                           -0.6458724585399134, -0.1528122416408308, 0.0, 0.6611281231681304, -0.704772385385899,
                           -0.2573042742623125, 0.0, 0.4259679219173819, 0.060157769686237685, 0.1221867316976898,
                           1.0]).reshape(4, 4).T
        if self.debug:
            print("timestamp_gray_image={} [ns]".format(gray_image.header.stamp.nsecs))
            print("timestamp_color_image={} [ns]".format(color_image.header.stamp.nsecs))
            print("timestamp_point_cloud={} [ns]".format(point_cloud.header.stamp.nsecs))
        gray_image = self.bridge.imgmsg_to_cv2(gray_image, desired_encoding='passthrough')
        color_image = self.bridge.imgmsg_to_cv2(color_image, desired_encoding='passthrough')
        color_image_copy = color_image.copy()
        depth_image = np.array(self.bridge.imgmsg_to_cv2(depth_image, desired_encoding='passthrough'), dtype=np.float32)
        depth_image_copy = depth_image.copy()
        if log_data:
            cv2.imwrite("/home/user/code/zed-sdk/mahdi/log/debug_chess_calibration_d/image_left_{}.jpeg".format(N),
                        color_image_copy)
            np.save("/home/user/code/zed-sdk/mahdi/log/debug_chess_calibration_d/image_left_{}.npy".format(N),
                    color_image_copy)
            cv2.imwrite("/home/user/code/zed-sdk/mahdi/log/debug_chess_calibration_d/depth_map_{}.jpeg".format(N),
                        depth_image_copy)
            np.save("/home/user/code/zed-sdk/mahdi/log/debug_chess_calibration_d/depth_map_{}.npy".format(N),
                    depth_image_copy)
        # tags = self.at_detector.detect(gray_image, False, camera_params=None)
        # tag_idx = 0
        # for tag in tags:
        #     for idx in range(len(tag.corners)):
        # cv2.line(color_image_copy, tuple(tag.corners[idx - 1, :].astype(int)),
        #          tuple(tag.corners[idx, :].astype(int)),
        #          (0, 255, 0))
        # cv2.drawMarker(color_image_copy, tuple(tag.corners[idx, :].astype(int)), color=(255, 0, 0))
        # cv2.putText(color_image_copy, str(idx),
        #             org=(tag.corners[idx, 0].astype(int) + 3, tag.corners[idx, 1].astype(int) + 3),
        #             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #             fontScale=0.5,
        #             color=(255, 0, 0))
        # cv2.line(depth_image_copy, tuple(tag.corners[idx - 1, :].astype(int)),
        #          tuple(tag.corners[idx, :].astype(int)),
        #          (0, 255, 0))
        # cv2.drawMarker(depth_image_copy, tuple(tag.corners[idx, :].astype(int)), color=(255, 0, 0))
        # cv2.putText(depth_image_copy, str(idx),
        #             org=(tag.corners[idx, 0].astype(int) + 3, tag.corners[idx, 1].astype(int) + 3),
        #             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #             fontScale=0.5,
        #             color=(255, 0, 0))
        # cv2.putText(color_image_copy, str(tag_idx),
        #             org=(tag.corners[0, 0].astype(int) - 5, tag.corners[0, 1].astype(int) - 5),
        #             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #             fontScale=1.5,
        #             color=(200, 0, 200))
        # x_p_obj_img, y_p_obj_img = tag.corners[3]
        # u = int(x_p_obj_img)
        # v = int(y_p_obj_img)
        # # intrinsic based
        # Z0 = np.nanmean(depth_image_copy[v - 2: v + 2, u - 2: u + 2])
        # X0 = Z0 * (u - self.cx) / self.fx
        # Y0 = Z0 * (v - self.cy) / self.fy
        # p_obj_ca = np.array([X0, Y0, Z0])
        # self.p_obj_ca.x = p_obj_ca[0]
        # self.p_obj_ca.y = p_obj_ca[1]
        # self.p_obj_ca.z = p_obj_ca[2]
        self.cv2_imshow(color_image_copy, window_name="left image")
        try:
            gray = cv2.cvtColor(color_image_copy, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
            # If found, add object points, image points (after refining them)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 300, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(color_image_copy, (9, 6), corners2, ret)
            for idx in range(0, 54):
                cv2.putText(color_image_copy, str(idx),
                            org=(corners2.squeeze()[idx, 0].astype(int), corners2.squeeze()[idx, 1].astype(int)),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.4,
                            color=(255, 0, 0))
            self.cv2_imshow(color_image_copy, window_name="left image")
        except:
            pass
        # self.cv2_imshow(depth_image_copy, window_name="depth image")
        if log_data:
            cv2.imwrite(
                "/home/user/code/zed-sdk/mahdi/log/debug_chess_calibration_d/image_left_marked_{}.jpeg".format(N),
                color_image_copy)
            np.save("/home/user/code/zed-sdk/mahdi/log/debug_chess_calibration_d/corners_chess_{}.npy".format(N),
                    corners2)
            np.save(
                "/home/user/code/zed-sdk/mahdi/log/debug_chess_calibration_d/Chessboard_detections_{}.npy".format(N),
                corners2)
            D0 = list(sensor_msgs.point_cloud2.read_points(point_cloud))
            D = np.asarray(D0).reshape(1200, 1920, 4)
            np.save("/home/user/code/zed-sdk/mahdi/log/debug_chess_calibration_d/ROS_point_cloud_{}.npy".format(N), D)
            # np.save("/home/user/code/zed-sdk/mahdi/log/debug_chess_calibration_d/Apriltag_detections_{}.npy".format(N),
            #         tag.corners)
            # O_T_EE = np.array([[0.0193126, -0.999226, -0.0339827, 0.248292],
            #                    [-0.961512, -0.0278786, 0.273313, -0.272983],
            #                    [-0.274049, 0.0273964, -0.961325, 0.298596],
            #                    [0, 0, 0, 1]])
            np.save("/home/user/code/zed-sdk/mahdi/log/debug_chess_calibration_d/O_T_EE_{}.npy".format(N), O_T_EE)
            print("data are saved.")


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
    # publish p_obj_ca, T_ftc_ca
    pub_p_obj_ca = rospy.Publisher('p_obj_ca', Vector3, queue_size=10)
    pub_T_ftc_ca = rospy.Publisher('T_ftc_ca', Float64MultiArray, queue_size=10)
    rate = rospy.Rate(1)  # 10hz
    while not rospy.is_shutdown():
        pub_p_obj_ca.publish(server.p_obj_ca)
        pub_T_ftc_ca.publish(server.T_ftc_ca)
        rate.sleep()

    rospy.spin()
