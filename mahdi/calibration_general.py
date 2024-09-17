import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cv2
import pyzed.sl as sl
import math
import numpy as np
import sys
import math
import cv2
from dt_apriltags import Detector

import rospy
from sensor_msgs.msg import Image, CameraInfo
import message_filters
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
from geometry_msgs.msg import Vector3Stamped
import time
from geometry_msgs.msg import Vector3Stamped


def cv2_imshow(a, window_name="image"):
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
    cv2.waitKey(500)
    cv2.destroyAllWindows()


def rectification(raw_l, raw_r, A_raw_l, A_raw_r):
    rec_l = None
    rec_r = None
    return rec_l, rec_r


def save_calib_data(log_dir):
    # Create a Camera object
    zed = sl.Camera()
    # roi = sl.Rect(42, 56, 120, 15)
    # zed.set_camera_settings_roi(sl.VIDEO_SETTINGS.AEC_AGC_ROI, roi, sl.SIDE.BOTH)
    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL  # Use ULTRA depth mode
    init_params.coordinate_units = sl.UNIT.MILLIMETER  # Use meter units (for depth measurements)
    # TODO check this
    init_params.camera_disable_self_calib = True
    init_params.depth_minimum_distance = 200  # Set the minimum depth perception distance
    init_params.depth_maximum_distance = 900  # Set the maximum depth perception distance
    # Open the camera
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:  # Ensure the camera has opened succesfully
        print("Camera Open : " + repr(status) + ". Exit program.")
        exit()
    cam_info = zed.get_camera_information()
    A_rect_r = cam_info.camera_configuration.calibration_parameters.right_cam
    A_rect_l = cam_info.camera_configuration.calibration_parameters.left_cam
    # Create and set RuntimeParameters after opening the camera
    runtime_parameters = sl.RuntimeParameters()
    id = 1
    # O_T_EE = np.array(
    #     [0.5857499126595401, -0.807897207019151, 0.06465206579224003, 0.0, -0.8095020378945257, -0.5792613229089174,
    #      0.09562174087324073, 0.0, -0.039802862550637655, -0.10834849142197248, -0.9933158292000374, 0.0,
    #      0.2847365754690908, -0.2245196045043414, 0.20950541682112178, 1.0]).reshape(4, 4).T
    image = sl.Mat()
    depth = sl.Mat()
    depth_conf = sl.Mat()
    depth_conf_img = sl.Mat()
    depth_img = sl.Mat()
    point_cloud = sl.Mat()
    # mirror_ref = sl.Transform()
    # # TODO what is this?
    # mirror_ref.set_translation(sl.Translation(2.75, 4.0, 0))
    # A new image is available if grab() returns SUCCESS
    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        # Retrieve left image
        zed.retrieve_image(image, sl.VIEW.LEFT)
        img_l = image.get_data()
        cv2_imshow(image.get_data(), window_name="left image")
        # # Retrieve right image
        # zed.retrieve_image(image, sl.VIEW.RIGHT)
        # cv2_imshow(image.get_data(), window_name="right image")
        # # # Retrieve side by side image
        # zed.retrieve_image(image, sl.VIEW.SIDE_BY_SIDE )
        # imS = cv2.resize(cv2.cvtColor(image.get_data(), cv2.COLOR_BGRA2RGBA), (1920, 1200))
        # cv2_imshow(imS, window_name="SIDE_BY_SIDE  image")
        # Retrieve depth map. Depth is aligned on the left image
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
        depth_l = depth.get_data()
        zed.retrieve_measure(depth_conf, sl.MEASURE.CONFIDENCE)
        # TODO
        depth_conf = depth_conf.get_data()
        zed.retrieve_image(depth_img, sl.VIEW.DEPTH)
        cv2_imshow(depth_img.get_data(), window_name="depth map")
        zed.retrieve_image(depth_conf_img, sl.VIEW.CONFIDENCE)
        cv2_imshow(depth_conf_img.get_data(), window_name="depth confidence map")
        # Close the camera
    cube_length = 36.2  # [mm]
    n_x = 9
    n_y = 6
    tmp = np.mgrid[0:n_x, 0:n_y].T.reshape(-1, 2) * np.array([1, -1])
    objectPoints1 = np.zeros((n_y * n_x, 3), np.float32)
    # objectPoints[:, 0] = tmp[:, 0] * cube_length
    # objectPoints[:, 2] = 0
    # objectPoints[:, 1] = tmp[:, 1] * cube_length
    # objectPoints[:, 0] = tmp[:, 0] * cube_length + (12.5 * 50 - 12.8 - 36.2)
    # objectPoints[:, 1] = -2.5 * 50 - 22.5 - 7
    # objectPoints[:, 2] = tmp[:, 1] * cube_length + 8.9 + 36.2 - 25
    objectPoints1[:, 0] = tmp[:, 0] * cube_length + (12.5 * 50 + 87.5 - 20.5 - cube_length) - 8 * cube_length
    objectPoints1[:, 1] = -3.5 * 50 - 20.7 - 6.9
    objectPoints1[:, 2] = tmp[:, 1] * cube_length - 25 + 38.2 + cube_length + 5 * cube_length
    objectPoints2 = np.zeros((n_x * n_y, 3), np.float32)
    objectPoints2[:, 0] = +3.5 * 50 + 20.7 + 4.5
    objectPoints2[:, 1] = tmp[:, 0] * cube_length + 1.5 * 50 + 23 - 101.8 - cube_length - 8 * cube_length
    objectPoints2[:, 2] = tmp[:, 1] * cube_length + 39 + cube_length - 25 + 5 * cube_length
    objectPoints = np.vstack((objectPoints1, objectPoints2))
    # cube_length = 18.5  # [mm]
    # tmp = np.mgrid[0:18, 0:11].T.reshape(-1, 2) * np.array([-1, 1])
    # objectPoints[:, 0] = tmp[:, 0] * cube_length + 5
    # objectPoints[:, 1] = -103.5
    # objectPoints[:, 2] = tmp[:, 1] * cube_length * 2 - 25 + 8.9
    try:
        gray = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (n_x, n_y), None)
        # If found, add object points, image points (after refining them)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 300, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(img_l, (n_x, n_y), corners2, ret)
        for idx in range(0, n_x * n_y):
            cv2.putText(img_l, str(idx),
                        org=(corners2.squeeze()[idx, 0].astype(int), corners2.squeeze()[idx, 1].astype(int)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.4,
                        color=(255, 0, 0))
        cv2_imshow(img_l, window_name="left image")
        imagePoints1 = corners2.reshape((n_x * n_y, 1, 2))
        # repeat fpr the second chessboard
        try:
            gray[:, 800:] = 0
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (n_x, n_y), None)
            # If found, add object points, image points (after refining them)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 300, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(img_l, (n_x, n_y), corners2, ret)
            for idx in range(0, n_x * n_y):
                cv2.putText(img_l, str(idx),
                            org=(corners2.squeeze()[idx, 0].astype(int), corners2.squeeze()[idx, 1].astype(int)),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.4,
                            color=(255, 0, 0))
            cv2_imshow(img_l, window_name="left image")
        except:
            pass
        imagePoints2 = corners2.reshape((n_x * n_y, 1, 2))
        imagePoints = np.vstack((imagePoints1, imagePoints2))
        objpts = []  # 3d point in real world space
        objpts.append(objectPoints)
        imgpts = []  # 3d point in real world space
        imgpts.append(corners2)
        cameraMatrix = np.array([[A_rect_l.fx, 0, A_rect_l.cx], [0, A_rect_l.fy, A_rect_l.cy], [0, 0, 1]])
        retval, r_t2c, t_t2c, reprojErr = cv2.solvePnPGeneric(objectPoints, imagePoints, cameraMatrix,
                                                              distCoeffs=A_rect_l.disto,
                                                              flags=cv2.SOLVEPNP_ITERATIVE)
        print("reprojErr=", reprojErr)
        # # # TODO
        # r_t2c = np.load(log_dir + "/r_t2c.npy")
        # t_t2c = np.load(log_dir + "/t_t2c.npy")
        # r_t2c = np.load(log_dir + "/r_t2c_1.npy")
        # t_t2c = np.load(log_dir + "/t_t2c_1.npy")
        H_t2c = np.vstack((np.hstack((cv2.Rodrigues(r_t2c[0])[0], t_t2c[0])), np.array([0, 0, 0, 1])))
        t_c2t = -np.matrix(cv2.Rodrigues(r_t2c[0])[0]).T * np.matrix(t_t2c[0])
        R_c2t = np.matrix(cv2.Rodrigues(r_t2c[0])[0]).T
        H_c2t = np.vstack((np.hstack((R_c2t, t_c2t.reshape(3, 1))),
                           np.array([0, 0, 0, 1])))
        err_all_c = []
        err_all_t = []
        for k in range(n_x * n_y):
            u = int(imagePoints[k, :, 0])
            v = int(imagePoints[k, :, 1])
            Z = np.nanmean(depth_l[v - 2: v + 2, u - 2: u + 2])
            X = Z * (u - A_rect_l.cx) / A_rect_l.fx
            Y = Z * (v - A_rect_l.cy) / A_rect_l.fy
            P_c = np.array([X, Y, Z, 1])
            P_t = np.append(objectPoints[k, :], 1)
            P_c_hat = np.matrix(H_t2c) * np.matrix(P_t.reshape(4, 1))
            # P_c_hat = H_t2c @ P_t.reshape(4,1)
            P_t_hat = np.matrix(H_c2t) * np.matrix(P_c.reshape(4, 1))
            err_c = P_c_hat[:3] - P_c[:3].reshape(3, 1)
            err_t = P_t_hat[:3] - P_t[:3].reshape(3, 1)
            # print("(***ONLY X and Z should be near zero) err_t[mm]=", err_t)
            # print("(***ONLY Y and Z should be near zero) err_t[mm]=", err_t)
            print("k={}\n".format(k))
            print("err_c[mm]=", err_c)
            print("norm(err_c)[mm]=", np.linalg.norm(err_c))
            err_all_c.append(np.linalg.norm(err_c))
            print("err_t[mm]=", err_t)
            print("norm(err_t)[mm]=", np.linalg.norm(err_t))
            err_all_t.append(np.linalg.norm(err_t))
        print("mean err_all_c[mm]", np.mean(err_all_c))
        print("mean err_all_t[mm]", np.mean(err_all_t))
    except:
        pass
    np.save(log_dir + "/r_t2c_{}.npy".format(str(id)), r_t2c)
    np.save(log_dir + "/t_t2c_{}.npy".format(str(id)), t_t2c)
    np.save(log_dir + "/depth_l_{}.npy".format(str(id)), depth_l)
    cv2.imwrite(log_dir + "/depth_conf_img_{}.jpeg".format(str(id)), depth_conf_img.get_data())
    np.save(log_dir + "/depth_conf_{}.npy".format(str(id)), depth_conf)
    cv2.imwrite(log_dir + "/depth_img_{}.jpeg".format(str(id)), depth_img.get_data())
    cv2.imwrite(log_dir + "/img_l_{}.jpeg".format(str(id)), img_l)
    np.save(log_dir + "/img_l_{}.npy".format(str(id)), img_l)
    np.save(log_dir + "/imagePoints_{}.npy".format(str(id)), imagePoints)
    np.save(log_dir + "/objectPoints_{}.npy".format(str(id)), objectPoints)
    np.save(log_dir + "/corners2_{}.npy".format(str(id)), corners2)
    A = np.array([A_rect_l.fx, A_rect_l.fy, A_rect_l.cx, A_rect_l.cy])
    np.save(log_dir + "/A_rect_l_{}.npy".format(str(id)), A)
    # np.save(log_dir + "/O_T_EE_{}.npy".format(str(id)), O_T_EE)
    zed.close()


def hand_to_eye_calib(log_dir):
    r_t2c = []
    t_t2c = []
    r_b2g = []
    t_b2g = []
    N = 5
    for k in range(1, N + 1):
        r_t2c_ = np.load(log_dir + "/r_t2c_{}.npy".format(str(k)))
        t_t2c_ = np.load(log_dir + "/t_t2c_{}.npy".format(str(k)))
        H_g2b = np.load(log_dir + "/O_T_EE_{}.npy".format(str(k)))

        r_t2c = np.append(r_t2c, r_t2c_.reshape(3, 1))
        t_t2c = np.append(t_t2c, t_t2c_.reshape(3, 1))

        R_g2b = H_g2b[0:3, 0:3]
        t_g2b = H_g2b[3, 0:3]

        r_b2g = np.append(r_b2g, cv2.Rodrigues(R_g2b.T)[0])
        t_b2g = np.append(t_b2g, -R_g2b.T @ t_g2b)
    r_t2c = r_t2c.reshape(3, N)
    t_t2c = t_t2c.reshape(3, N)
    r_b2g = r_b2g.reshape(3, N)
    t_b2g = t_b2g.reshape(3, N)
    r_c2b, t_c2b = cv2.calibrateHandEye(r_b2g.T, t_b2g.T, r_t2c.T, t_t2c.T, method=cv2.CALIB_HAND_EYE_TSAI)
    print("r_c2b=", r_c2b)
    print("t_c2b=", t_c2b)


def check_accuracy(log_dir):
    id = 1
    zed = sl.Camera()
    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL  # Use ULTRA depth mode
    init_params.coordinate_units = sl.UNIT.MILLIMETER  # Use meter units (for depth measurements)
    # TODO check this
    init_params.camera_disable_self_calib = True
    init_params.depth_minimum_distance = 200  # Set the minimum depth perception distance
    init_params.depth_maximum_distance = 900  # Set the maximum depth perception distance
    # Open the camera(after you call the open here then the rectification is applied)
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:  # Ensure the camera has opened succesfully
        print("Camera Open : " + repr(status) + ". Exit program.")
        exit()
    cam_info = zed.get_camera_information()
    # Create and set RuntimeParameters after opening the camera
    runtime_parameters = sl.RuntimeParameters()
    image = sl.Mat()
    depth = sl.Mat()
    depth_conf = sl.Mat()
    depth_conf_img = sl.Mat()
    depth_img = sl.Mat()
    point_cloud = sl.Mat()
    # A new image is available if grab() returns SUCCESS
    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        # Retrieve left image
        zed.retrieve_image(image, sl.VIEW.LEFT)
        img_l = image.get_data()
        cv2_imshow(img_l, window_name="left image")
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
        depth_l = depth.get_data()
        zed.retrieve_measure(depth_conf, sl.MEASURE.CONFIDENCE)
    at_detector = Detector(searchpath=['apriltags'],
                           families='tag36h11',
                           nthreads=1,
                           quad_decimate=1.0,
                           quad_sigma=0.0,
                           refine_edges=1,
                           decode_sharpening=0.25,
                           debug=0)
    # load calibration data
    r_t2c = np.load(log_dir + "/r_t2c_{}.npy".format(str(id)))
    t_t2c = np.load(log_dir + "/t_t2c_{}.npy".format(str(id)))
    A = np.load(log_dir + "/A_rect_l_{}.npy".format(str(id)))
    gray = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    tags = at_detector.detect(gray, False, camera_params=None)
    for tag in tags:
        for idx in range(len(tag.corners)):
            # print(
            #     "!!corner detected on image plane location = ({},{}) [pxls].".format(
            #         tag.corners[idx, 0], tag.corners[idx, 1]))
            cv2.line(img_l, tuple(tag.corners[idx - 1, :].astype(int)),
                     tuple(tag.corners[idx, :].astype(int)),
                     (0, 255, 0))
            cv2.drawMarker(img_l, tuple(tag.corners[idx, :].astype(int)), color=(255, 0, 0))
            cv2.putText(img_l, str(idx),
                        org=(tag.corners[idx, 0].astype(int) + 3, tag.corners[idx, 1].astype(int) + 3),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(255, 0, 0))
            cv2.line(img_l, tuple(tag.corners[idx - 1, :].astype(int)),
                     tuple(tag.corners[idx, :].astype(int)),
                     (0, 255, 0))
            cv2.drawMarker(img_l, tuple(tag.corners[idx, :].astype(int)), color=(255, 0, 0))
            cv2.putText(img_l, str(idx),
                        org=(tag.corners[idx, 0].astype(int) + 3, tag.corners[idx, 1].astype(int) + 3),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(255, 0, 0))
        cv2_imshow(img_l, window_name="left image + Apriltag detections")
        H_t2c = np.vstack((np.hstack((cv2.Rodrigues(r_t2c[0])[0], t_t2c[0])), np.array([0, 0, 0, 1])))
        t_c2t = -np.matrix(cv2.Rodrigues(r_t2c[0])[0]).T * np.matrix(t_t2c[0])
        R_c2t = np.matrix(cv2.Rodrigues(r_t2c[0])[0]).T
        H_c2t = np.vstack((np.hstack((R_c2t, t_c2t.reshape(3, 1))),
                           np.array([0, 0, 0, 1])))
        err_all_c = []
        err_all_t = []
        # for idx_tag in range(4):
        idx_tag = 3
        u, v = tag.corners[idx_tag]
        u = int(u)
        v = int(v)
        Z = np.nanmean(depth_l[v - 2: v + 2, u - 2: u + 2])
        X = Z * (u - A[2]) / A[0]
        Y = Z * (v - A[3]) / A[1]
        P_c = np.array([X, Y, Z, 1])
        # manually measure
        P_t = np.append(np.array([10.5 * 50 - 30.5 + 4.8 + 0.5, -4.5 * 50 - 15, -25 + 76.4 + 4.8]), 1)
        P_c_hat = np.matrix(H_t2c) * np.matrix(P_t.reshape(4, 1))
        # P_c_hat = H_t2c @ P_t.reshape(4,1)
        P_t_hat = np.matrix(H_c2t) * np.matrix(P_c.reshape(4, 1))
        err_c = P_c_hat[:3] - P_c[:3].reshape(3, 1)
        err_t = P_t_hat[:3] - P_t[:3].reshape(3, 1)
        print("idx_tag={}\n".format(idx_tag))
        print("err_c[mm]=", err_c)
        print("norm(err_c)[mm]=", np.linalg.norm(err_c))
        err_all_c.append(np.linalg.norm(err_c))
        print("err_t[mm]=", err_t)
        print("norm(err_t)[mm]=", np.linalg.norm(err_t))
        err_all_t.append(np.linalg.norm(err_t))
        print("mean err_all_c[mm]", np.mean(err_all_c))
        print("mean err_all_t[mm]", np.mean(err_all_t))
        print("ended.")


class ROSserver:
    def __init__(self):
        self.img_gray = None
        self.t = 0
        self.t0 = 0
        self.helper_index = 0
        self.id = 0

        self.myMessage = Vector3Stamped()
        self.myMessage.vector.x = 0
        self.myMessage.vector.y = 0
        self.myMessage.vector.z = 0
        self.myMessage.header.stamp = rospy.Time.now()

        self.N = 100  # total number of images to save
        self.color_image = np.zeros((self.N, 1200, 1920, 4))
        self.depth_map = np.zeros((self.N, 1200, 1920))
        self.depth_confidence_map = np.zeros((self.N, 1200, 1920))
        self.t = np.zeros(self.N)

    def gotdata(self, color_image, depth_map, depth_confidence_map, camera_info):
        if self.id < self.N:
            if self.id == 0:
                # print("+++++++++++++++++++++++++++")
                self.t0 = color_image.header.stamp.secs + color_image.header.stamp.nsecs / 10 ** 9
                np.save(log_dir + "/t0.npy", self.t0)
                fx = camera_info.K[0]
                fy = camera_info.K[4]
                cx = camera_info.K[2]
                cy = camera_info.K[5]
                self.A = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                np.save(log_dir + "/A.npy", self.A)
            # if self.id == 20:  # switch on the conveyor after at data 20
            #     self.myMessage.vector.x = 1
            # if np.sum(self.A != np.array(
            #         [[camera_info.K[0], 0, camera_info.K[2]], [0, camera_info.K[4], camera_info.K[5]], [0, 0, 1]])) > 0:
            #     raise ("error: intrinsic camera matrix changed!")
            # print("t-t0[ms]=", (color_image.header.stamp.secs+color_image.header.stamp.nsecs/10**9 - self.t0)*1000)
            # print("dt[ms]=", (color_image.header.stamp.secs + color_image.header.stamp.nsecs / 10 ** 9 - self.t) * 1000)
            t = color_image.header.stamp.secs + color_image.header.stamp.nsecs / 10 ** 9
            # print("timestamp_color_image={} [s]".format(
            #     color_image.header.stamp.secs + color_image.header.stamp.nsecs / 10 ** 9))
            # print("timestamp_depth_map={} [s]".format(
            #     depth_map.header.stamp.secs + depth_map.header.stamp.nsecs / 10 ** 9))
            # print("timestamp_depth_confidence_map={} [s]".format(
            #     depth_confidence_map.header.stamp.secs + depth_confidence_map.header.stamp.nsecs / 10 ** 9))
            color_image = CvBridge().imgmsg_to_cv2(color_image, desired_encoding='passthrough')
            depth_map = np.array(CvBridge().imgmsg_to_cv2(depth_map, desired_encoding='passthrough'), dtype=np.float32)
            depth_confidence_map = np.array(
                CvBridge().imgmsg_to_cv2(depth_confidence_map, desired_encoding='passthrough'), dtype=np.float32)
            self.color_image[self.id,:,:,:]=color_image
            self.depth_map[self.id,:,:]=depth_map
            self.depth_confidence_map[self.id,:,:]=depth_confidence_map
            self.t[self.id]=t
            # print("self.id=", self.id)
        self.id += 1
        if self.id == self.N:
            for id_ in range(0,self.N):
                np.save(log_dir + "/color_image_{}.npy".format(str(id_)), self.color_image[id_,:,:,:])
                cv2.imwrite(log_dir + "/color_image_{}.png".format(str(id_)), self.color_image[id_,:,:,:])
                np.save(log_dir + "/depth_map_{}.npy".format(str(id_)), self.depth_map[id_,:,:])
                cv2.imwrite(log_dir + "/depth_map_{}.png".format(str(id_)), self.depth_map[id_,:,:])
                np.save(log_dir + "/depth_confidence_map_{}.npy".format(str(id_)), self.depth_confidence_map[id_,:,:])
                cv2.imwrite(log_dir + "/depth_confidence_map_{}.png".format(str(id_)), self.depth_confidence_map[id_,:,:])
                np.save(log_dir + "/time_stamp_{}.npy".format(str(id_)), self.t[id_])


def get_timestamped_ros_data(log_dir):
    rospy.init_node('my_node')
    server = ROSserver()
    color_image_listener = message_filters.Subscriber('/zedxm/zed_node/rgb/image_rect_color', Image)
    depth_map_listener = message_filters.Subscriber('/zedxm/zed_node/depth/depth_registered', Image)
    camera_info_listener = message_filters.Subscriber('/zedxm/zed_node/depth/camera_info', CameraInfo)
    depth_confidence_map_listener = message_filters.Subscriber('/zedxm/zed_node/confidence/confidence_map', Image)
    ts = message_filters.ApproximateTimeSynchronizer(
        [color_image_listener, depth_map_listener, depth_confidence_map_listener, camera_info_listener],
        1,
        0.001)  # slop parameter in the constructor that defines the delay (in seconds) with which messages can be synchronized.
    ts.registerCallback(server.gotdata)
    # rospy.spin()

    # nodeName = "messagepublisher"
    # topicName = "information"
    # rospy.init_node(nodeName, anonymous=True)
    # publisher = rospy.Publisher(topicName, Vector3Stamped, queue_size=0)
    hz = 1000
    rate = rospy.Rate(hz)
    # id_seq=0
    while not rospy.is_shutdown():
        # id_seq += 1
        # server.myMessage.header.seq = id_seq
        # rospy.loginfo(server.myMessage)
        # publisher.publish(server.myMessage)
        rate.sleep()

    # rate = rospy.Rate(1)  # 10hz
    # while not rospy.is_shutdown():
    #     rate.sleep()


class fast_ROSserver:
    def __init__(self):

        self.img_gray = None
        self.tVec = 0
        self.helper_index = 0
        self.id = 0
        self.A = np.zeros((3, 3))
        self.at_detector = Detector(searchpath=['apriltags'],
                                    families='tag36h11',
                                    nthreads=1,
                                    quad_decimate=1.0,
                                    quad_sigma=0.0,
                                    refine_edges=1,
                                    decode_sharpening=0.25,
                                    debug=0)
        self.corners = np.zeros((1, 4, 2))
        self.P_c_hat = np.zeros((4, 1))
        self.conf_Z = 0

        self.myMessage = Vector3Stamped()
        self.myMessage.vector.x = 0
        self.myMessage.vector.y = 0
        self.myMessage.vector.z = 0
        self.myMessage.header.stamp = rospy.Time.now()

    def gotdata(self, color_image, depth_map, depth_confidence_map, camera_info):
        # N = 60  # total number of images to save
        # if self.id < N:
        ti = time.time()
        if self.id == 0:
            # self.t0 = color_image.header.stamp.secs + color_image.header.stamp.nsecs / 10 ** 9
            # np.save(log_dir + "/t0.npy", self.t0)
            fx = camera_info.K[0]
            fy = camera_info.K[4]
            cx = camera_info.K[2]
            cy = camera_info.K[5]
            self.A = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            # np.save(log_dir + "/A.npy", self.A)
        if self.id == 10:  # switch on the conveyor after at data 10
            self.myMessage.vector.x = 1
        if np.sum(self.A != np.array(
                [[camera_info.K[0], 0, camera_info.K[2]], [0, camera_info.K[4], camera_info.K[5]], [0, 0, 1]])) > 0:
            raise ("error: intrinsic camera matrix changed!")
        self.tVec = np.append(self.tVec, color_image.header.stamp.secs + color_image.header.stamp.nsecs / 10 ** 9)
        # print("1[ms]", (time.time() - ti) * 1000)
        # t = time.time()
        color_image = CvBridge().imgmsg_to_cv2(color_image, desired_encoding='passthrough')
        # ROI=[[1, 0.5], [1, 1], [0.55, 1], [0.55, 0.5]]
        u0 = int(0.55 * 1920)
        v0 = int(0.5 * 1200)
        color_image = color_image[v0:1200, u0:1920, :]
        # cv2.imwrite(log_dir + "/color_image.jpeg", color_image)
        # print("2[ms]", (time.time() - t) * 1000)
        # t = time.time()
        # TODO subscribe directly to gray image?
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        # print("2gray[ms]", (time.time() - t) * 1000)
        # t = time.time()
        # TODO here is time consuming
        tags = self.at_detector.detect(gray, False, camera_params=None)
        tag = tags[0]
        idx_tag = 3
        u, v = tag.corners[idx_tag]
        # u,v=[1217.89733887,  720.61071777]
        u = int(u) + u0
        v = int(v) + v0
        # print("3[ms]", (time.time() - t) * 1000)
        # t = time.time()
        depth_map = np.array(CvBridge().imgmsg_to_cv2(depth_map, desired_encoding='passthrough'), dtype=np.float32)
        # A = np.nan_to_num(depth_map, nan=0, posinf=0, neginf=0)*255
        # cv2.imwrite(log_dir + "/depth_map.jpeg", A)
        # print("4[ms]", (time.time() - t) * 1000)
        # t = time.time()
        Z = np.nanmean(depth_map[v - 2: v + 2, u - 2: u + 2])
        X = Z * (u - self.A[0, 2]) / self.A[0, 0]
        Y = Z * (v - self.A[1, 2]) / self.A[1, 1]
        self.P_c_hat = np.hstack((self.P_c_hat, np.array([[X], [Y], [Z], [1]])))
        # print("5[ms]", (time.time() - t) * 1000)
        # t = time.time()
        depth_confidence_map = np.array(
            CvBridge().imgmsg_to_cv2(depth_confidence_map, desired_encoding='passthrough'), dtype=np.float32)
        self.conf_Z = np.append(self.conf_Z, np.nanmean(depth_confidence_map[v - 2: v + 2, u - 2: u + 2]))
        self.id += 1
        # print("6[ms]", (time.time() - t) * 1000)
        # t = time.time()
        if self.id == 160:
            print("self.id=", self.id)
            np.save(log_dir + "/tVec_s3.npy", self.tVec)
            np.save(log_dir + "/P_c_hat_s3.npy", self.P_c_hat)
            np.save(log_dir + "/conf_Z_s3.npy", self.conf_Z)
            exit()
        #     print("self.t=", self.tVec)
        #     print("dt[ms]=", np.diff(self.tVec[1:]) * 1000)
        # print("7[ms]", (time.time() - t) * 1000)
        # print("8[ms]", (time.time() - ti) * 1000)


def fast_get_timestamped_ros_data(log_dir):
    rospy.init_node('my_node')
    server = fast_ROSserver()
    color_image_listener = message_filters.Subscriber('/zedxm/zed_node/rgb/image_rect_color', Image)
    depth_map_listener = message_filters.Subscriber('/zedxm/zed_node/depth/depth_registered', Image)
    camera_info_listener = message_filters.Subscriber('/zedxm/zed_node/depth/camera_info', CameraInfo)
    depth_confidence_map_listener = message_filters.Subscriber('/zedxm/zed_node/confidence/confidence_map', Image)
    ts = message_filters.ApproximateTimeSynchronizer(
        [color_image_listener, depth_map_listener, depth_confidence_map_listener, camera_info_listener],
        5,
        0.01)  # slop parameter in the constructor that defines the delay (in seconds) with which messages can be synchronized.
    ts.registerCallback(server.gotdata)
    topicName = "information"
    publisher = rospy.Publisher(topicName, Vector3Stamped, queue_size=0)
    hz = 100
    rate = rospy.Rate(hz)
    while not rospy.is_shutdown():
        publisher.publish(server.myMessage)
        rate.sleep()


def calc_results(log_dir):
    N = 100
    P_c_all = []
    P_t_hat_all = []
    P_t_gt_all = np.zeros((N, 4))
    time_stamp_all = np.zeros(N)
    for id in range(0, N):
        print(id)
        img_l = np.load(log_dir + "/color_image_{}.npy".format(str(id))).astype(dtype="uint8")
        time_stamp = np.load(log_dir + "/time_stamp_{}.npy".format(str(id)))
        depth_l = np.load(log_dir + "/depth_map_{}.npy".format(str(id)))

        # load calibration data
        r_t2c = np.load("/home/user/code/zed-sdk/mahdi/log/hand_to_eye_calibration/two_t_on_table/r_t2c_1.npy")
        t_t2c = np.load("/home/user/code/zed-sdk/mahdi/log/hand_to_eye_calibration/two_t_on_table/t_t2c_1.npy")
        A = np.load(log_dir + "/A.npy")
        gray = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        at_detector = Detector(searchpath=['apriltags'],
                               families='tag36h11',
                               nthreads=1,
                               quad_decimate=1.0,
                               quad_sigma=0.0,
                               refine_edges=1,
                               decode_sharpening=0.25,
                               debug=0)
        tags = at_detector.detect(gray, False, camera_params=None)
        # for tag in tags:
        tag = tags[0]
        if False:
            for idx in range(len(tag.corners)):
                # print(
                #     "!!corner detected on image plane location = ({},{}) [pxls].".format(
                #         tag.corners[idx, 0], tag.corners[idx, 1]))
                cv2.line(img_l, tuple(tag.corners[idx - 1, :].astype(int)),
                         tuple(tag.corners[idx, :].astype(int)),
                         (0, 255, 0))
                cv2.drawMarker(img_l, tuple(tag.corners[idx, :].astype(int)), color=(255, 0, 0))
                cv2.putText(img_l, str(idx),
                            org=(tag.corners[idx, 0].astype(int) + 3, tag.corners[idx, 1].astype(int) + 3),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            color=(255, 0, 0))
                cv2.line(img_l, tuple(tag.corners[idx - 1, :].astype(int)),
                         tuple(tag.corners[idx, :].astype(int)),
                         (0, 255, 0))
                cv2.drawMarker(img_l, tuple(tag.corners[idx, :].astype(int)), color=(255, 0, 0))
                cv2.putText(img_l, str(idx),
                            org=(tag.corners[idx, 0].astype(int) + 3, tag.corners[idx, 1].astype(int) + 3),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            color=(255, 0, 0))
            cv2_imshow(img_l, window_name="left image + Apriltag detections id={}".format(str(id)))
        # H_t2c = np.vstack((np.hstack((cv2.Rodrigues(r_t2c[0])[0], t_t2c[0])), np.array([0, 0, 0, 1])))
        t_c2t = -np.matrix(cv2.Rodrigues(r_t2c[0])[0]).T * np.matrix(t_t2c[0])
        R_c2t = np.matrix(cv2.Rodrigues(r_t2c[0])[0]).T
        H_c2t = np.vstack((np.hstack((R_c2t, t_c2t.reshape(3, 1))),
                           np.array([0, 0, 0, 1])))
        # err_all_c = []
        # err_all_t = []
        # for idx_tag in range(4):
        idx_tag = 3
        u, v = tag.corners[idx_tag]
        u = int(u)
        v = int(v)
        Z = np.nanmean(depth_l[v - 2: v + 2, u - 2: u + 2]) * 1000
        X = Z * (u - A[0, 2]) / A[0, 0]
        Y = Z * (v - A[1, 2]) / A[1, 1]
        P_c_hat = np.array([X, Y, Z, 1])
        P_c_all = np.hstack((P_c_all, P_c_hat))
        time_stamp_all[id] = time_stamp
        # # manually measure
        # P_t = np.append(np.array([10.5 * 50 - 30.5 + 4.8 + 0.5, -4.5 * 50 - 15, -25 + 76.4 + 4.8]), 1)
        # P_c_hat = np.matrix(H_t2c) * np.matrix(P_t.reshape(4, 1))
        # # P_c_hat = H_t2c @ P_t.reshape(4,1)
        P_t_hat = np.matrix(H_c2t) * np.matrix(P_c_hat.reshape(4, 1))
        P_t_hat_all = np.hstack((P_t_hat_all, np.asarray(P_t_hat).squeeze()))

        # err_c = P_c_hat[:3] - P_c[:3].reshape(3, 1)
        # err_t = P_t_hat[:3] - P_t[:3].reshape(3, 1)
        # print("idx_tag={}\n".format(idx_tag))
        # print("err_c[mm]=", err_c)
        # print("norm(err_c)[mm]=", np.linalg.norm(err_c))
        # err_all_c.append(np.linalg.norm(err_c))
        # print("err_t[mm]=", err_t)
        # print("norm(err_t)[mm]=", np.linalg.norm(err_t))
        # err_all_t.append(np.linalg.norm(err_t))
        # print("mean err_all_c[mm]", np.mean(err_all_c))
        # print("mean err_all_t[mm]", np.mean(err_all_t))
    P_c_all = P_c_all.reshape(N, 4)
    P_t_hat_all = P_t_hat_all.reshape(N, 4)
    dt = np.diff(time_stamp_all, n=1, axis=-1) * 1000
    t = (time_stamp_all - time_stamp_all[0]) * 1000
    dPc = np.diff(P_c_all, n=1, axis=0)
    norm_dPc = np.linalg.norm(dPc[:, :3], axis=1)

    plt.figure(figsize=(8, 12))
    plt.subplot(411)
    plt.plot(t[:-1] / 1000, norm_dPc, '-bo')
    plt.ylabel("norm dPc [mm]")
    plt.subplot(412)
    plt.plot(t[:-1] / 1000, dPc[:, 0], '-bo')
    plt.ylabel("dPc[0] [mm]")
    plt.subplot(413)
    plt.plot(t[:-1] / 1000, dPc[:, 1], '-bo')
    plt.ylabel("dPc[1] [mm]")
    plt.subplot(414)
    plt.plot(t[:-1] / 1000, dPc[:, 2], '-bo')
    plt.ylabel("dPc[2] [mm]")
    plt.xlabel("time [s]")
    plt.savefig(log_dir + "/dPc.png", format="png")
    plt.show()

    # P_t_gt_0_measured = np.array([506.5, -342, 77.5])
    # P_t_gt_0_measured = np.array([9.5*50+20-1+11.3+5, -(3.5*50+65+95), 76.8-25+3+5])
    # P_t_gt_0_measured = np.array([9.5 * 50 + 30 + 4 + 0.1 + 4.3, -5.5 * 50 + 4 + 0.1, 80 - 25 - 1.5 + 14.1])
    P_t_gt_0_measured = np.array([513.4, -270.9, 68])
    # TODO
    N_trigger = 20
    bias = np.array(
        [np.nanmean(P_t_hat_all[1:N_trigger, 0]) - P_t_gt_0_measured[0],
         np.nanmean(P_t_hat_all[1:N_trigger - 1, 1]) - P_t_gt_0_measured[1],
         np.nanmean(P_t_hat_all[1:N_trigger, 2]) - P_t_gt_0_measured[2]])
    # bias = np.array([0, 0, 0])
    P_t_gt_0 = P_t_gt_0_measured + bias
    P_t_gt_all = np.zeros((N, 3)) + P_t_gt_0
    # P_t_gt_all[:, 1] += np.linalg.norm(P_c_all[:, :3], axis=1) - np.linalg.norm(P_c_all[:, :3], axis=1)[0]
    for i in range(N_trigger, 87):
        P_t_gt_all[i, 1] += 2.  # correcting initial position to move
        for j in range(N_trigger, i):
            P_t_gt_all[i, 1] += (dt[j]) * 34.9028e-3
    for k in range(87, N):
        P_t_gt_all[k, 1] = P_t_gt_all[k - 1, 1]
    # TODO
    # dp=np.diff(time_stamp_all) * 21.64*2
    # for i in range(N_trigger,N):
    #     P_t_gt_all[i, 1]=np.sum(dp[N_trigger:i+1])
    # P_t_gt_all[:, 1] += np.linalg.norm(P_c_all[:,:3]-P_c_all[0,:3], axis=1)
    plt.figure(figsize=(8, 10))
    # fig.suptitle('speed_1', fontsize=10)
    plt.subplot(411)
    plt.title("speed = 34.9028 [mm/s]; nominal_pwm_period=25*2=50 micSec")
    plt.plot(t / 1000, np.linalg.norm(P_t_hat_all[:, :3] - P_t_gt_all, axis=1), '-ko')
    plt.ylabel("norm Pt_hat-Pt_gt [mm]")
    plt.subplot(412)
    plt.plot(t / 1000, P_t_hat_all[:, 0], '-bo', label="Pt_hat")
    plt.plot(t / 1000, P_t_gt_all[:, 0], '-ro', label="Pt_gt")
    plt.ylabel("Pt[0] [mm]")
    plt.subplot(413)
    plt.plot(t / 1000, P_t_hat_all[:, 1], '-bo', label="Pt_hat")
    plt.plot(t / 1000, P_t_gt_all[:, 1], '-ro', label="Pt_gt")
    plt.ylabel("Pt[1] [mm]")
    plt.subplot(414)
    plt.plot(t / 1000, P_t_hat_all[:, 2], '-bo', label="Pt_hat")
    plt.plot(t / 1000, P_t_gt_all[:, 2], '-ro', label="Pt_gt")
    plt.ylabel("Pt[2] [mm]")
    plt.xlabel("time [s]")
    plt.legend(loc="upper right")
    plt.savefig(log_dir + "/Pt.png", format="png")
    plt.show()

    plt.figure(figsize=(8, 12))
    plt.subplot(411)
    plt.plot(t[:-1] / 1000, norm_dPc / dt * 1000, '-bo')
    plt.ylabel("norm dPc/dt [mm/s]")
    plt.subplot(412)
    plt.plot(t[:-1] / 1000, dPc[:, 0] / dt * 1000, '-bo')
    plt.ylabel("dPc[0]/dt [mm/s]")
    plt.subplot(413)
    plt.plot(t[:-1] / 1000, dPc[:, 1] / dt * 1000, '-bo')
    plt.ylabel("dPc[1]/dt [mm/s]")
    plt.subplot(414)
    plt.plot(t[:-1] / 1000, dPc[:, 2] / dt * 1000, '-bo')
    plt.ylabel("dPc[2] [mm/s]")
    plt.xlabel("time [s]")
    plt.savefig(log_dir + "/speed_est_camera.png", format="png")
    plt.show()

    np.save(log_dir + "/P_t_gt_0_measured.npy", P_t_gt_0_measured)
    np.save(log_dir + "/P_t_gt_all.npy", P_t_gt_all)
    np.save(log_dir + "/P_t_hat_all.npy", P_t_hat_all)
    np.save(log_dir + "/t.npy", t)
    print("ended.")


def fast_calc_results(log_dir):
    tVec = np.load(log_dir + "/tVec_s3.npy")[1:]
    P_c_hat = np.load(log_dir + "/P_c_hat_s3.npy")[:, 1:] * 1000  # [mm]
    conf_Z = np.load(log_dir + "/conf_Z_s3.npy")[1:]
    tVec = ((tVec - tVec[0]) * 1000)  # in [ms]
    plt.figure(figsize=(8, 12))
    plt.subplot(411)
    plt.plot(tVec, P_c_hat[0, :], '-bo')
    plt.ylabel("P_c_hat[0] [mm]")
    plt.subplot(412)
    plt.plot(tVec, P_c_hat[1, :], '-bo')
    plt.ylabel("P_c_hat[1] [mm]")
    plt.subplot(413)
    plt.plot(tVec, P_c_hat[2, :], '-bo')
    plt.ylabel("P_c_hat[2] [mm]")
    plt.subplot(414)
    plt.plot(tVec, conf_Z, '-r*')
    plt.ylabel("conf_Z")
    plt.xlabel("t [ms]")
    plt.savefig(log_dir + "/measurements_s3.png", format="png")
    plt.show()


class publish_measurement_server:
    def __init__(self):
        self.start_measurement = False
        self.img_gray = None
        self.tVec = 0
        self.helper_index = 0
        self.id = 0
        self.A = np.zeros((3, 3))
        self.at_detector = Detector(searchpath=['apriltags'],
                                    families='tag36h11',
                                    nthreads=1,
                                    quad_decimate=1.0,
                                    quad_sigma=0.0,
                                    refine_edges=1,
                                    decode_sharpening=0.25,
                                    debug=0)
        self.corners = np.zeros((1, 4, 2))
        self.P_c_hat = np.zeros((4, 1))
        self.conf_Z = 0

        self.myMessage = Vector3Stamped()
        self.myMessage.vector.x = 0
        self.myMessage.vector.y = 0
        self.myMessage.vector.z = 0
        self.myMessage.header.stamp = rospy.Time.now()
        self.pub_p_hat_w = rospy.Publisher('p_hat_w', Vector3Stamped, queue_size=10)
        r_t2c = np.load("/home/user/code/zed-sdk/mahdi/log/hand_to_eye_calibration/two_t_on_table/r_t2c_1.npy")
        t_t2c = np.load("/home/user/code/zed-sdk/mahdi/log/hand_to_eye_calibration/two_t_on_table/t_t2c_1.npy")
        t_c2t = -np.matrix(cv2.Rodrigues(r_t2c[0])[0]).T * np.matrix(t_t2c[0])
        R_c2t = np.matrix(cv2.Rodrigues(r_t2c[0])[0]).T
        self.H_c2t = np.vstack((np.hstack((R_c2t, t_c2t.reshape(3, 1))),
                                np.array([0, 0, 0, 1])))
        self.P_t_hat = Vector3Stamped()

    def gotdata(self, color_image, depth_map, camera_info):
        fx = camera_info.K[0]
        fy = camera_info.K[4]
        cx = camera_info.K[2]
        cy = camera_info.K[5]
        A = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        self.t = color_image.header.stamp.secs + color_image.header.stamp.nsecs
        color_image = CvBridge().imgmsg_to_cv2(color_image, desired_encoding='passthrough')
        u0 = int(0.55 * 1920)
        v0 = int(0.5 * 1200)
        color_image = color_image[v0:1200, u0:1920, :]
        # TODO subscribe directly to gray image?
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        # TODO here is time consuming
        tags = self.at_detector.detect(gray, False, camera_params=None)
        # print("tags=",tags)
        tag = tags[0]
        idx_tag = 3
        u, v = tag.corners[idx_tag]
        u = int(u) + u0
        v = int(v) + v0
        depth_map = np.array(CvBridge().imgmsg_to_cv2(depth_map, desired_encoding='passthrough'), dtype=np.float32)
        Z = np.nanmean(depth_map[v - 2: v + 2, u - 2: u + 2]) * 1000  # [mm]
        X = Z * (u - A[0, 2]) / A[0, 0]  # [mm]
        Y = Z * (v - A[1, 2]) / A[1, 1]  # [mm]
        P_c_hat = np.array([[X], [Y], [Z], [1]])
        P_t_hat = np.matrix(self.H_c2t) * np.matrix(P_c_hat.reshape(4, 1))
        self.P_t_hat.vector.x = P_t_hat[0]
        self.P_t_hat.vector.y = P_t_hat[1]
        self.P_t_hat.vector.z = P_t_hat[2]

        self.pub_p_hat_w.publish(self.P_t_hat)
        # info = "measurement published!!!"
        info = "self.P_t_hat.vector.x-y-z={},{},{}".format(self.P_t_hat.vector.x,self.P_t_hat.vector.y,self.P_t_hat.vector.z)
        rospy.loginfo(info)
    def gotTriggerData(self, trigger):
        if trigger.vector.x == 1:
            color_image_listener = message_filters.Subscriber('/zedxm/zed_node/rgb/image_rect_color', Image)
            depth_map_listener = message_filters.Subscriber('/zedxm/zed_node/depth/depth_registered', Image)
            camera_info_listener = message_filters.Subscriber('/zedxm/zed_node/depth/camera_info', CameraInfo)
            # trigger_listener = message_filters.Subscriber('PRIMITIVE_velocity_controller/STEPPERMOTOR_messages', Vector3Stamped)
            ts = message_filters.ApproximateTimeSynchronizer(
                [color_image_listener, depth_map_listener, camera_info_listener],
                100,
                0.01)  # slop parameter in the constructor that defines the delay (in seconds) with which messages can be synchronized.
            ts.registerCallback(self.gotdata)
            hz = 100
            rate = rospy.Rate(hz)
            while not rospy.is_shutdown():
                rate.sleep()
            # rospy.spin()


    # def gotTriggerData(self, data):
    #     info = "STEPPERMOTOR_messages received!"
    #     rospy.loginfo(info)
    #     if self.start_measurement == False:
    #         if data.vector.x == 1:
    #             self.start_measurement = True
    #             info = "Trigger measurement received!!"
    #             rospy.loginfo(info)


def publish_measurement():
    rospy.init_node('my_node')
    server = publish_measurement_server()
    # color_image_listener = message_filters.Subscriber('/zedxm/zed_node/rgb/image_rect_color', Image)
    # depth_map_listener = message_filters.Subscriber('/zedxm/zed_node/depth/depth_registered', Image)
    # camera_info_listener = message_filters.Subscriber('/zedxm/zed_node/depth/camera_info', CameraInfo)
    # # trigger_listener = message_filters.Subscriber('PRIMITIVE_velocity_controller/STEPPERMOTOR_messages', Vector3Stamped)
    # ts = message_filters.ApproximateTimeSynchronizer(
    #     [color_image_listener, depth_map_listener, camera_info_listener],
    #     100,
    #     0.01)  # slop parameter in the constructor that defines the delay (in seconds) with which messages can be synchronized.
    # ts.registerCallback(server.gotdata)
    # topicName = "information"
    topicName = "PRIMITIVE_velocity_controller/STEPPERMOTOR_messages"
    subscriber = rospy.Subscriber(topicName, Vector3Stamped, server.gotTriggerData, queue_size=5)
    hz = 100
    rate = rospy.Rate(hz)
    while not rospy.is_shutdown():
        rate.sleep()
    # rospy.spin()


if __name__ == '__main__':
    log_dir = "/home/user/code/zed-sdk/mahdi/log/hand_to_eye_calibration/two_t_on_table/validation/test_vel_34_9028_mm_s_faster_slop_1ms"
    # save_calib_data(log_dir)
    # hand_to_eye_calib(log_dir)
    # check_accuracy(log_dir)
    # get_timestamped_ros_data(log_dir)
    calc_results(log_dir)
    # fast_get_timestamped_ros_data(log_dir)
    # fast_calc_results(log_dir)
    # publish_measurement()
