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
    cv2.waitKey(0)
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

    def gotdata(self, color_image, depth_map, depth_confidence_map, camera_info):
        N=252 #total number of images to save
        if self.id < N:
            if self.helper_index == 0:
                # print("+++++++++++++++++++++++++++")
                self.t0 = color_image.header.stamp.secs + color_image.header.stamp.nsecs / 10 ** 9
                np.save(log_dir + "/t0.npy", self.t0)
                self.helper_index += 1
                fx = camera_info.K[0]
                fy = camera_info.K[4]
                cx = camera_info.K[2]
                cy = camera_info.K[5]
                self.A = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                np.save(log_dir + "/A.npy", self.A)
            if np.sum(self.A != np.array([[camera_info.K[0], 0, camera_info.K[2]], [0, camera_info.K[4], camera_info.K[5]], [0, 0, 1]]))>0:
                raise ("error: intrinsic camera matrix changed!")
            # print("t-t0[ms]=", (color_image.header.stamp.secs+color_image.header.stamp.nsecs/10**9 - self.t0)*1000)
            print("dt[ms]=", (color_image.header.stamp.secs + color_image.header.stamp.nsecs / 10 ** 9 - self.t) * 1000)
            self.t = color_image.header.stamp.secs + color_image.header.stamp.nsecs / 10 ** 9
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
            np.save(log_dir + "/color_image_{}.npy".format(str(self.id)), color_image)
            cv2.imwrite(log_dir + "/color_image_{}.png".format(str(self.id)), color_image)
            np.save(log_dir + "/depth_map_{}.npy".format(str(self.id)), depth_map)
            cv2.imwrite(log_dir + "/depth_map_{}.png".format(str(self.id)), depth_map)
            np.save(log_dir + "/depth_confidence_map_{}.npy".format(str(self.id)), depth_confidence_map)
            cv2.imwrite(log_dir + "/depth_confidence_map_{}.png".format(str(self.id)), depth_confidence_map)
            np.save(log_dir + "/time_stamp_{}.npy".format(str(self.id)), self.t)
            print("self.id=", self.id)
        self.id += 1


def get_timestamped_ros_data(log_dir):
    rospy.init_node('my_node')
    server = ROSserver()
    color_image_listener = message_filters.Subscriber('/zedxm/zed_node/rgb/image_rect_color', Image)
    depth_map_listener = message_filters.Subscriber('/zedxm/zed_node/depth/depth_registered', Image)
    camera_info_listener = message_filters.Subscriber('/zedxm/zed_node/depth/camera_info', CameraInfo)
    depth_confidence_map_listener = message_filters.Subscriber('/zedxm/zed_node/confidence/confidence_map', Image)
    ts = message_filters.ApproximateTimeSynchronizer(
        [color_image_listener, depth_map_listener, depth_confidence_map_listener, camera_info_listener],
        5,
        0.01)  # slop parameter in the constructor that defines the delay (in seconds) with which messages can be synchronized.
    ts.registerCallback(server.gotdata)
    # rate = rospy.Rate(1)  # 10hz
    # while not rospy.is_shutdown():
    #     rate.sleep()
    rospy.spin()


def approximate_speed(log_dir):
    id=0
    img_l=np.load(log_dir + "/color_image_{}.npy".format(str(id)))
    depth_l=np.load(log_dir + "/depth_map_{}.npy".format(str(id)))

    # load calibration data
    r_t2c = np.load("/home/user/code/zed-sdk/mahdi/log/hand_to_eye_calibration/two_t_on_table/r_t2c_1.npy")
    t_t2c = np.load("/home/user/code/zed-sdk/mahdi/log/hand_to_eye_calibration/two_t_on_table/t_t2c_1.npy")
    A = np.load(log_dir + "/A_rect_l_{}.npy".format(str(id)))
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

if __name__ == '__main__':
    log_dir = "/home/user/code/zed-sdk/mahdi/log/hand_to_eye_calibration/two_t_on_table/validation/speed_1"
    # save_calib_data(log_dir)
    # hand_to_eye_calib(log_dir)
    # check_accuracy(log_dir)
    get_timestamped_ros_data(log_dir)
    # approximate_speed(log_dir)