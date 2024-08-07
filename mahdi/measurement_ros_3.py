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

# R_O2c = cv2.Rodrigues(np.array([1.57035026, 0.22209125,-0.21333207]))[0]
# t_O2c = np.array([-0.31389315, 0.09370776, 0.61622412])
#cv2 Intrinsic calibrated matrix [[711.81239653   0.         887.91336336], [  0.         730.16768199 614.32765277], [  0.           0.           1.        ]]
# R_O2c = np.load("/home/user/code/zed-sdk/mahdi/log/debug_chess_calibration_d/R_target2cam.npy")
# t_O2c = np.load("/home/user/code/zed-sdk/mahdi/log/debug_chess_calibration_d/t_target2cam.npy")
# R_c2O=R_O2c.transpose()
# t_c2O=np.matmul(-R_O2c.T,t_O2c)
H_c2t_2 = np.load("/home/user/code/zed-sdk/mahdi/log/debug_chess_calibration_d/H_c2t_2.npy")
R_c2O=H_c2t_2[:3,:3]
t_c2O=H_c2t_2[:3,3]
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
        self.p_obj_O = Vector3()
        # self.cx = 479.4909973144531
        # self.cy = 299.3664855957031
        # self.fx = 367.72650146484375
        # self.fy = 367.6239929199219

        # # _b from /zedxm/zed_node/rgb/camera_info
        # self.cx = 472.77484130859375
        # self.cy = 281.85357666015625
        # self.fx = 377.4441223144531
        # self.fy = 377.4441223144531

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

        # # _f from /zedxm/zed_node/depth/camera_info
        # self.cx = 487.22515869140625
        # self.cy = 258.14642333984375
        # self.fx = 377.4441223144531
        # self.fy = 377.

        # # _f from /zedxm/zed_node/depth/camera_info
        # # ROS dept/camera_info fullHD no truncation
        # self.cx = 974.4503173828125
        # self.cy = 516.2928466796875
        # self.fx = 754.8882446289062
        # self.fy = 754.8882446289062

        # # _f from /zedxm/zed_node/depth/camera_info
        # # ROS dept/camera_info HD1200 no truncation
        # self.cx = 974.64697265625
        # self.cy = 575.6713256835938
        # self.fx = 747.6171875
        # self.fy = 747.6171875

        # ROS dept/camera_info HD1200 no truncation(TODO WHY DIFFERENT FROM ABOVE WITH SAME RUN OF ROS ZED?!!!! it is supposed to be as _f above!!!!)
        self.cx = 945.35302734375
        self.cy = 624.3286743164062
        self.fx = 747.6171875
        self.fy = 747.6171875


        self.debug = 0

        # self.pub_p_obj_ca = rospy.Publisher('p_obj_ca', Vector3, queue_size=10)
        # self.pub_T_ftc_ca = rospy.Publisher('T_ftc_ca', Float64MultiArray, queue_size=10)
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
        tag_idx = 1
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
            # x_t_ftc2_img, y_t_ftc2_img = (tag.corners[0] + tag.corners[3]) / 2
            # u = int(x_t_ftc2_img)
            # v = int(y_t_ftc2_img)
            # us = np.linspace(u - 2, u + 2, 5, dtype=int)
            # vs = np.linspace(v - 2, v + 2, 5, dtype=int)
            # uvs = np.array(np.meshgrid(us, vs)).reshape(2, 25).transpose().tolist()
            # D0 = list(sensor_msgs.point_cloud2.read_points(point_cloud, uvs=uvs))
            # D = np.nanmean(np.asarray(D0), 0)
            # X = D[0]
            # Y = D[1]
            # Z = D[2]
            # check_idx_tag = np.array([X, Y, Z])
            # # if check_idx_tag[0]<0.18:
            # #     tag_idx = 0
            # # else:
            # #     tag_idx =1
            tag_idx = 0  # TODO
            if self.debug:
                cv2.putText(color_image_copy, str(tag_idx),
                            org=(tag.corners[0, 0].astype(int) - 5, tag.corners[0, 1].astype(int) - 5),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1.5,
                            color=(200, 0, 200))
            # if tag_idx == 0:  # TODO make this and else conditions robust to order of tags
            #     x_t_ftc2_img, y_t_ftc2_img = (tag.corners[0] + tag.corners[3]) / 2
            #     # arrayPosition = y_t_ftc2_img * point_cloud.row_step + x_t_ftc2_img * point_cloud.point_step
            #     # arrayPosX = arrayPosition + point_cloud.fields[0].offset
            #     # arrayPosY = arrayPosition + point_cloud.fields[1].offset
            #     # arrayPosZ = arrayPosition + point_cloud.fields[2].offset
            #     # X = point_cloud.data[int(arrayPosX)]
            #     # Y = point_cloud.data[int(arrayPosY)]
            #     # Z = point_cloud.data[int(arrayPosZ)]
            #     # # TODO check u and v
            #     # v = x_t_ftc2_img
            #     # u = y_t_ftc2_img
            #     # depth_tmp = depth_image_copy[int(u) - 3:int(u) + 3, int(v) - 3:int(v) + 3]
            #     # # TODO 0.250 [m] prior magic number
            #     # acceptable_idx = depth_tmp < 0.250
            #     # Z = np.mean(depth_tmp[acceptable_idx])
            #     # X = Z * (u - self.cx) / self.fx
            #     # Y = Z * (v - self.cy) / self.fy
            #     u = int(x_t_ftc2_img)
            #     v = int(y_t_ftc2_img)
            #     us = np.linspace(u - 2, u + 2, 5, dtype=int)
            #     vs = np.linspace(v - 2, v + 2, 5, dtype=int)
            #     uvs= np.array(np.meshgrid(us, vs)).reshape(2, 25).transpose().tolist()
            #     D0 = list(sensor_msgs.point_cloud2.read_points(point_cloud, uvs=uvs))
            #     D = np.nanmean(np.asarray(D0), 0)
            #     X = D[0]
            #     Y = D[1]
            #     Z = D[2]
            #     t_ftc2_ca = np.array([X, Y, Z])
            #     if self.debug:
            #         print("t_ftc2_ca = {} [m].".format(t_ftc2_ca))
            #     x_y_ftc2_img, y_y_ftc2_img = tag.corners[3]
            #     print("----------",tag.corners)
            #     # arrayPosition = y_y_ftc2_img * point_cloud.row_step + x_y_ftc2_img * point_cloud.point_step
            #     # arrayPosX = arrayPosition + point_cloud.fields[0].offset
            #     # arrayPosY = arrayPosition + point_cloud.fields[1].offset
            #     # arrayPosZ = arrayPosition + point_cloud.fields[2].offset
            #     # X1 = point_cloud.data[int(arrayPosX)]
            #     # Y1 = point_cloud.data[int(arrayPosY)]
            #     # Z1 = point_cloud.data[int(arrayPosZ)]
            #     # for point in sensor_msgs.point_cloud2.read_points(point_cloud, skip_nans=True):
            #     #     pt_x = point[0]
            #     #     pt_y = point[1]
            #     #     pt_z = point[2]
            #     # for point in sensor_msgs.point_cloud2.read_points(point_cloud, skip_nans=True,
            #     #                                                   uvs=[int(x_y_ftc2_img), int(y_y_ftc2_img)]):
            #     #     pt_x = point[0]
            #     #     pt_y = point[1]
            #     #     pt_z = point[2]
            #     dpix=0
            #     # # # TODO check u and v
            #     # u = int(x_y_ftc2_img)
            #     # v = int(y_y_ftc2_img)
            #     # depth_tmp = depth_image_copy[v,u]
            #     # # TODO 0.250 [m] prior magic number
            #     # acceptable_idx = depth_tmp < 0.550
            #     # # Z=depth_image_copy[int(u),int(v)]
            #     # Z0 = np.mean(depth_tmp[acceptable_idx])
            #     # Z0=depth_image_copy
            #     # X0=np.zeros(np.shape(Z0))
            #     # Y0=np.zeros(np.shape(Z0))
            #     # for u in range(0,959):
            #     #     for v in range(0,539):
            #     #         X0[v,u]=Z0[v,u]* (u - self.cx) / self.fx
            #     #         Y0[v,u]=Z0[v,u]* (v - self.cy) / self.fy
            #     # D0 = list(sensor_msgs.point_cloud2.read_points(point_cloud))
            #     # D = np.asarray(D0).reshape(540,960,4)
            #     # X = D[:,:,0]
            #     # Y = D[:,:,1]
            #     # Z = D[:,:,2]
            #     # np.save("/home/user/code/zed-sdk/mahdi/log/X0_f.npy", X0)
            #     # np.save("/home/user/code/zed-sdk/mahdi/log/Y0_f.npy", Y0)
            #     # np.save("/home/user/code/zed-sdk/mahdi/log/Z0_f.npy", Z0)
            #     # np.save("/home/user/code/zed-sdk/mahdi/log/X_f.npy", X)
            #     # np.save("/home/user/code/zed-sdk/mahdi/log/Y_f.npy", Y)
            #     # np.save("/home/user/code/zed-sdk/mahdi/log/Z_f.npy", Z)
            #
            #     # X0 = Z0 * (u - self.cx) / self.fx
            #     # Y0 = Z0 * (v - self.cy) / self.fy
            #
            #     u = int(x_y_ftc2_img)
            #     v = int(y_y_ftc2_img)
            #     us = np.linspace(u - dpix, u + dpix, 2*dpix+1, dtype=int)
            #     vs = np.linspace(v - dpix, v + dpix, 2*dpix+1, dtype=int)
            #     uvs= np.array(np.meshgrid(us, vs)).reshape(2, (2*dpix+1)**2).transpose().tolist()
            #     D0 = list(sensor_msgs.point_cloud2.read_points(point_cloud, uvs=uvs))
            #     D = np.nanmean(np.asarray(D0), 0)
            #     X = D[0]
            #     Y = D[1]
            #     Z = D[2]
            #     y_ftc2_ca = np.array([X, Y, Z])
            #
            #     if self.debug:
            #         print("y_ftc2_ca = {} [m].".format(y_ftc2_ca))
            #     x_c_tag_img, y_c_tag_img = np.mean(tag.corners, 0)
            #     # arrayPosition = y_c_tag_img * point_cloud.row_step + x_c_tag_img * point_cloud.point_step
            #     # arrayPosX = arrayPosition + point_cloud.fields[0].offset
            #     # arrayPosY = arrayPosition + point_cloud.fields[1].offset
            #     # arrayPosZ = arrayPosition + point_cloud.fields[2].offset
            #     # X = point_cloud.data[int(arrayPosX)]
            #     # Y = point_cloud.data[int(arrayPosY)]
            #     # Z = point_cloud.data[int(arrayPosZ)]
            #     # # TODO check u and v
            #     # v = x_c_tag_img
            #     # u = y_c_tag_img
            #     # depth_tmp = depth_image_copy[int(u) - 3:int(u) + 3, int(v) - 3:int(v) + 3]
            #     # # TODO 0.250 [m] prior magic number
            #     # acceptable_idx = depth_tmp < 0.250
            #     # Z = np.mean(depth_tmp[acceptable_idx])
            #     # X = Z * (u - self.cx) / self.fx
            #     # Y = Z * (v - self.cy) / self.fy
            #     u = int(x_c_tag_img)
            #     v = int(y_c_tag_img)
            #     us = np.linspace(u - 2, u + 2, 5, dtype=int)
            #     vs = np.linspace(v - 2, v + 2, 5, dtype=int)
            #     uvs= np.array(np.meshgrid(us, vs)).reshape(2, 25).transpose().tolist()
            #     D0 = list(sensor_msgs.point_cloud2.read_points(point_cloud, uvs=uvs))
            #     D = np.nanmean(np.asarray(D0), 0)
            #     X = D[0]
            #     Y = D[1]
            #     Z = D[2]
            #     c_tag_ca = np.array([X, Y, Z])
            #     if self.debug:
            #         # print("y_c_tag_img = {} [m].".format(y_c_tag_img))
            #         print("c_tag_ca = {} [m].".format(c_tag_ca))
            #     # cx=479.491
            #     # cy=299.366
            #     # fx=367.726
            #     # fy=367.623
            #     # # TODO check u and v
            #     # v=x_c_tag_img
            #     # u=y_c_tag_img
            #     # depth_image = self.bridge.imgmsg_to_cv2(depth_image, desired_encoding='passthrough')
            #     # z=depth_image[int(u),int(v)]
            #     # x=depth_image[int(u),int(v)]*(u-self.cx/self.fx)
            #     # y=depth_image[int(u),int(v)]*(v-self.cy/self.fy)
            #     # print("(x,y,z)=",x,y,z)
            #     z_ftc2_ca = t_ftc2_ca + (t_ftc2_ca - c_tag_ca)
            #     x_ftc2_ca = np.cross(y_ftc2_ca, z_ftc2_ca)
            #     if self.debug:
            #         print("x_ftc2_ca_atFTC2 = {} [m].".format((x_ftc2_ca-t_ftc2_ca)/ np.linalg.norm(x_ftc2_ca - t_ftc2_ca)))
            #         print("y_ftc2_ca_atFTC2 = {} [m].".format((y_ftc2_ca-t_ftc2_ca)/ np.linalg.norm(y_ftc2_ca - t_ftc2_ca)))
            #         print("z_ftc2_ca_atFTC2 = {} [m].".format((z_ftc2_ca-t_ftc2_ca)/ np.linalg.norm(z_ftc2_ca - t_ftc2_ca)))
            #         print("t_ftc2_ca = {} [m].".format(t_ftc2_ca))
            #
            #     # R = np.vstack(((x_ftc2_ca - t_ftc2_ca) / np.linalg.norm((x_ftc2_ca - t_ftc2_ca)),
            #     #                (y_ftc2_ca - t_ftc2_ca) / np.linalg.norm((y_ftc2_ca - t_ftc2_ca)),
            #     #                (z_ftc2_ca - t_ftc2_ca) / np.linalg.norm((z_ftc2_ca - t_ftc2_ca)))).T
            # # with Apriltags 10 data at /home/user/code/zed-sdk/mahdi/log/debug_calibration
            # R_cam2gripper = np.array([[0.0069347, 0.69100592, -0.72281584],
            #                           [-0.99994363, -0.00101968, -0.01056827],
            #                           [-0.00803978, 0.72284838, 0.6909599]])
            # t_cam2gripper = np.array([[0.11904731],
            #                           [0.02483592],
            #                           [-0.10425902]])

            # # with Chessboard 10 data at /home/user/code/zed-sdk/mahdi/log/debug_chess_calibration_b
            # R_cam2gripper = np.array([[-0.00715975, 0.68714525, -0.72648478],
            #                           [-0.99971785, -0.0213735, -0.01036357],
            #                           [-0.0226488, 0.7262056, 0.6871044]])
            # t_cam2gripper = np.array([[0.1266802],
            #                           [0.02387603],
            #                           [-0.11579095]])
            # # with Chessboard 10 data at /home/user/code/zed-sdk/mahdi/log/debug_chess_calibration_c
            # R_cam2gripper = np.array([[0.42521763, 0.82658076, -0.36871969],
            #                           [-0.8736435, 0.48129405, 0.07143579],
            #                           [0.23651004, 0.29175381, 0.92679163]])
            # t_cam2gripper = np.array([[0.09649874],
            #                           [0.06223238],
            #                           [-0.03354414]])
            # # with Chessboard 16 data at /home/user/code/zed-sdk/mahdi/log/debug_chess_calibration_c and _b
            # R_cam2gripper = np.array([[-0.01229468, 0.68217078, -0.73108951],
            #                           [-0.99963352, -0.02602032, -0.00746848],
            #                           [-0.02411796, 0.73072976, 0.68224068]])
            # t_cam2gripper = np.array([[0.17828708],
            #                           [0.01306143],
            #                           [-0.1986957]])
            # # with Chessboard 6 data at /home/user/code/zed-sdk/mahdi/log/debug_chess_calibration_c just data 2 to 7
            # R_cam2gripper = np.array([[-0.04080051, 0.68714119, -0.72537735],
            #                           [-0.99886221, -0.04599016, 0.01261737],
            #                           [-0.02469031, 0.72506682, 0.68823579]])
            # t_cam2gripper = np.array([[0.11971944],
            #                           [0.01862274],
            #                           [-0.12272205]])
            # # with Chessboard 10 data at /home/user/code/zed-sdk/mahdi/log/debug_chess_calibration_c just data acceptable_id={2,3,4,5,6,7,10,11,12,14}
            # R_cam2gripper = np.array([[-0.03757553, 0.68510434, -0.72747517],
            #                           [-0.99897794, -0.0440552, 0.01010992],
            #                           [-0.02512271, 0.72711154, 0.68605952]])
            # t_cam2gripper = np.array([[0.12252526],
            #                           [0.01927766],
            #                           [-0.11923062]])
            # # with Chessboard 10 data at /home/user/code/zed-sdk/mahdi/log/debug_chess_calibration_c just data acceptable_id={2,3,4,5,6,7,10,11,12,14,15,16,17,18}
            # R_cam2gripper = np.array([[-0.03377383, 0.68655549, -0.72629256],
            #                           [-0.99900388, -0.04439721, 0.00448722],
            #                           [-0.02916464, 0.72572064, 0.68737106]])
            # t_cam2gripper = np.array([[0.12659985],
            #                           [0.01983556],
            #                           [-0.10860643]])
            # with Chessboard 10 data at /home/user/code/zed-sdk/mahdi/log/debug_chess_calibration_c just data acceptable_id={2,3,4,5,6,7,10,11,12,14,15,16,17}
            R_cam2gripper = np.array([[-0.03674008, 0.68441354, -0.72816775],
                                      [-0.99900686, -0.04353508, 0.00948627],
                                      [-0.02520831, 0.7277931, 0.68533331]])
            t_cam2gripper = np.array([[0.12300563],
                                      [0.01928199],
                                      [-0.1190063]])


            T_ftc_ca = np.vstack((np.hstack((R_cam2gripper, t_cam2gripper.reshape(3, 1))), np.array([0, 0, 0, 1])))
            self.T_ftc_ca.data = T_ftc_ca.flatten()
            # elif tag_idx == 1:
            # get position of edge of apriltag in the corner of robot table
            x_p_obj_img, y_p_obj_img = tag.corners[3]
            # arrayPosition = y_p_obj_img * point_cloud.row_step + x_p_obj_img * point_cloud.point_step
            # arrayPosX = arrayPosition + point_cloud.fields[0].offset
            # arrayPosY = arrayPosition + point_cloud.fields[1].offset
            # arrayPosZ = arrayPosition + point_cloud.fields[2].offset
            # X = point_cloud.data[int(arrayPosX)]
            # Y = point_cloud.data[int(arrayPosY)]
            # Z = point_cloud.data[int(arrayPosZ)]
            # # TODO check u and v
            # v = x_p_obj_img
            # u = y_p_obj_img
            # depth_tmp = depth_image_copy[int(u) - 3:int(u) + 3, int(v) - 3:int(v) + 3]
            # # TODO 0.500 [m] prior magic number
            # acceptable_idx = depth_tmp < 0.500
            # Z = np.mean(depth_tmp[acceptable_idx])
            # X = Z * (u - self.cx) / self.fx
            # Y = Z * (v - self.cy) / self.fy
            u = int(x_p_obj_img)
            v = int(y_p_obj_img)

            # # ROS point cloud based
            # us = np.linspace(u - 2, u + 2, 5, dtype=int)
            # vs = np.linspace(v - 2, v + 2, 5, dtype=int)
            # uvs = np.array(np.meshgrid(us, vs)).reshape(2, 25).transpose().tolist()
            # D0 = list(sensor_msgs.point_cloud2.read_points(point_cloud, uvs=uvs))
            # D = np.nanmean(np.asarray(D0), 0)
            # # X = D[0]
            # # Y = D[1]
            # # Z = D[2]
            # # ax.scatter(Z0, -X0, -Y0, color="blue")
            # # ax.scatter(X, Y, Z, color="red")
            # X = -D[1]
            # Y = -D[2]
            # Z = D[0]
            # print("-----np.array([X, Y, Z])=", np.array([X, Y, Z]))

            # intrinsic based
            Z0 = np.nanmean(depth_image_copy[v - 2: v + 2, u - 2: u + 2])
            X0 = Z0 * (u - self.cx) / self.fx
            Y0 = Z0 * (v - self.cy) / self.fy
            p_obj_ca = np.array([X0, Y0, Z0])
            # print("+++++np.array([X0, Y0, Z0]))=", np.array([X0, Y0, Z0]))
            # np.save("/home/user/code/zed-sdk/mahdi/log/marqlev_calib/p_c_3.npy", p_obj_ca)

            self.p_obj_ca.x = p_obj_ca[0]
            self.p_obj_ca.y = p_obj_ca[1]
            self.p_obj_ca.z = p_obj_ca[2]

            p_obj_O=np.matmul(p_obj_ca, R_c2O) + t_c2O.squeeze()
            self.p_obj_O.x = p_obj_O[0]
            self.p_obj_O.y = p_obj_O[1]
            self.p_obj_O.z = p_obj_O[2]
            if self.debug:
                print("p_obj_ca = {} [m].".format(p_obj_ca))
                print("p_obj_O = {} [m].".format(p_obj_O))
        if 0:
            self.cv2_imshow(color_image_copy, window_name="left image")
            self.cv2_imshow(depth_image_copy, window_name="depth image")
            # save 10 data for calibration
            cv2.imwrite("/home/user/code/zed-sdk/mahdi/log/debug_chess_calibration_b/left_image_1.jpeg",
                        color_image_copy)
            gray = cv2.cvtColor(color_image_copy, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)
            # If found, add object points, image points (after refining them)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            print("corners2=", corners2)
            np.save("/home/user/code/zed-sdk/mahdi/log/debug_chess_calibration_b/corners_chese_1.npy", corners2)
            if 0:
                np.save("/home/user/code/zed-sdk/mahdi/log/debug_chess_calibration_b/left_image_1.npy",
                        color_image_copy)
                cv2.imwrite("/home/user/code/zed-sdk/mahdi/log/debug_chess_calibration_b/depth_map_1.jpeg",
                            depth_image_copy)
                np.save("/home/user/code/zed-sdk/mahdi/log/debug_chess_calibration_b/depth_map_1.npy", depth_image_copy)
                D0 = list(sensor_msgs.point_cloud2.read_points(point_cloud))
                # D = np.asarray(D0).reshape(540, 960, 4)
                D = np.asarray(D0).reshape(1080, 1920, 4)
                np.save("/home/user/code/zed-sdk/mahdi/log/debug_chess_calibration_b/point_cloud_1.npy", D)
                np.save("/home/user/code/zed-sdk/mahdi/log/debug_chess_calibration_b/2Ddetections_1.npy", tag.corners)
                print("=============================")
                O_T_EE = np.array([[0.0193126, -0.999226, -0.0339827, 0.248292],
                                   [-0.961512, -0.0278786, 0.273313, -0.272983],
                                   [-0.274049, 0.0273964, -0.961325, 0.298596],
                                   [0, 0, 0, 1]])
                np.save("/home/user/code/zed-sdk/mahdi/log/debug_chess_calibration_b/O_T_EE_1.npy", O_T_EE)
                print("hi")
        # self.pub_p_obj_ca.publish(self.p_obj_ca)
        # self.pub_T_ftc_ca.publish(self.T_ftc_ca)
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

    # publish p_obj_ca, T_ftc_ca
    pub_p_obj_ca = rospy.Publisher('p_obj_ca', Vector3, queue_size=10)
    pub_p_obj_O = rospy.Publisher('p_obj_O', Vector3, queue_size=10)
    pub_T_ftc_ca = rospy.Publisher('T_ftc_ca', Float64MultiArray, queue_size=10)
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
        pub_p_obj_O.publish(server.p_obj_O)
        pub_T_ftc_ca.publish(server.T_ftc_ca)
        if 0:
            info = "p_obj_ca.x={}".format(server.p_obj_ca.x)
            rospy.loginfo(info)
            info = "p_obj_ca.y={}".format(server.p_obj_ca.y)
            rospy.loginfo(info)
            info = "p_obj_ca.z={}".format(server.p_obj_ca.z)
            rospy.loginfo(info)
            info = "T_ftc_ca.data={}".format(server.T_ftc_ca.data)
            rospy.loginfo(info)
        if True:
            info = "p_obj_O.x[mm]={}".format(server.p_obj_O.x*1000)
            rospy.loginfo(info)
            info = "p_obj_O.y[mm]={}".format(server.p_obj_O.y*1000)
            rospy.loginfo(info)
            info = "p_obj_O.z[mm]={}".format(server.p_obj_O.z*1000)
            rospy.loginfo(info)
        # try:
        #     info = "T_ftc_ca.data[1]={}".format(server.T_ftc_ca.data[0])
        #     rospy.loginfo(info)
        # except:
        #     pass
        rate.sleep()

    rospy.spin()
