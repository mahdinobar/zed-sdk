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


if __name__ == '__main__':
    # conf=np.load("/home/user/code/zed-sdk/mahdi/log/debug_calib_100/depth_conf_1.npy")
    log_dir = "/home/user/code/zed-sdk/mahdi/log/hand_to_eye_calibration"
    # Create a Camera object
    zed = sl.Camera()
    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL  # Use ULTRA depth mode
    init_params.coordinate_units = sl.UNIT.MILLIMETER  # Use meter units (for depth measurements)
    # TODO check this
    init_params.camera_disable_self_calib = False
    init_params.depth_minimum_distance = 100  # Set the minimum depth perception distance
    init_params.depth_maximum_distance = 600  # Set the maximum depth perception distance
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
    zed.close()

    cube_length = 36.2  # [mm]
    tmp = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * np.array([-1, 1])
    objectPoints1 = np.zeros((6 * 9, 3), np.float32)
    objectPoints1[:, 0] = tmp[:, 0] * cube_length + (12.5 * 50 - 12.8 - 36.2)
    objectPoints1[:, 1] = -2.5 * 50 - 22.5 - 7
    objectPoints1[:, 2] = tmp[:, 1] * cube_length + 8.9 + 36.2 - 25

    objectPoints2 = np.zeros((6 * 9, 3), np.float32)
    objectPoints2[:, 1] = tmp[:, 0] * cube_length - (25 - 3 + 36.2)
    objectPoints2[:, 0] = +4.5 * 50 + 4
    objectPoints2[:, 2] = tmp[:, 1] * cube_length + 8.9 + 36.2 - 25

    objectPoints = np.vstack((objectPoints1, objectPoints2))

    # cube_length = 18.5  # [mm]
    # tmp = np.mgrid[0:18, 0:11].T.reshape(-1, 2) * np.array([-1, 1])
    # objectPoints[:, 0] = tmp[:, 0] * cube_length + 5
    # objectPoints[:, 1] = -103.5
    # objectPoints[:, 2] = tmp[:, 1] * cube_length * 2 - 25 + 8.9

    try:
        gray = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
        # If found, add object points, image points (after refining them)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 300, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(img_l, (9, 6), corners2, ret)
        for idx in range(0, 54):
            cv2.putText(img_l, str(idx),
                        org=(corners2.squeeze()[idx, 0].astype(int), corners2.squeeze()[idx, 1].astype(int)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.4,
                        color=(255, 0, 0))
        cv2_imshow(img_l, window_name="left image")
    except:
        pass

    imagePoints1 = corners2.reshape((54, 1, 2))
    # repeat fpr the second chessboard
    try:
        gray[:, 800:] = 0
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
        # If found, add object points, image points (after refining them)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 300, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(img_l, (9, 6), corners2, ret)
        for idx in range(0, 54):
            cv2.putText(img_l, str(idx),
                        org=(corners2.squeeze()[idx, 0].astype(int), corners2.squeeze()[idx, 1].astype(int)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.4,
                        color=(255, 0, 0))
        cv2_imshow(img_l, window_name="left image")
    except:
        pass
    imagePoints2 = corners2.reshape((54, 1, 2))
    imagePoints = np.vstack((imagePoints1, imagePoints2))
    objpts = []  # 3d point in real world space
    objpts.append(objectPoints)
    imgpts = []  # 3d point in real world space
    imgpts.append(corners2)
    cameraMatrix = np.array([[A_rect_l.fx, 0, A_rect_l.cx], [0, A_rect_l.fy, A_rect_l.cy], [0, 0, 1]])

    retval, r_t2c, t_t2c, reprojErr = cv2.solvePnPGeneric(objectPoints, imagePoints, cameraMatrix,
                                                          distCoeffs=A_rect_l.disto,
                                                          flags=cv2.SOLVEPNP_ITERATIVE)
    print("reprojErr=",reprojErr)


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
    for k in range(108):
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

    O_T_EE = np.array([0.058487896153253345, -0.2934875842206725, 0.9541619101667643, 0.0, -0.7479825753354957,
                       -0.6458724585399134, -0.1528122416408308, 0.0, 0.6611281231681304, -0.704772385385899,
                       -0.2573042742623125, 0.0, 0.4259679219173819, 0.060157769686237685, 0.1221867316976898,
                       1.0]).reshape(4, 4).T
    np.save(log_dir+"/r_t2c_1.npy", r_t2c)
    np.save(log_dir+"/t_t2c_1.npy", t_t2c)
    np.save(log_dir + "/depth_l_1.npy", depth_l)
    cv2.imwrite(log_dir + "/depth_conf_img_1.jpeg", depth_conf_img.get_data())
    np.save(log_dir + "/depth_conf_1.npy", depth_conf)
    cv2.imwrite(log_dir + "/depth_img_1.jpeg", depth_img.get_data())
    cv2.imwrite(log_dir + "/img_l_1.jpeg", img_l)
    np.save(log_dir + "/img_l_1.npy", img_l)
    np.save(log_dir + "/imagePoints_1.npy", imagePoints)
    np.save(log_dir + "/objectPoints_1.npy", objectPoints)
    np.save(log_dir + "/corners2_1.npy", corners2)

