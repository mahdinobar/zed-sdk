import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cv2

if __name__ == '__main__':
    # termination criteria
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # fname = '/home/user/code/zed-sdk/mahdi/log/debug_chess_calibration/left_image_3.jpeg'
    # img = cv2.imread(fname)
    # gray = cv2.cvtColor(img, cv2.COLOR_cGR2GRAY)
    # Find the chess board corners
    # ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)
    # If found, add object points, image points (after refining them)
    # corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    # np.save("/home/user/code/zed-sdk/mahdi/log/debug_chess_calibration/corners_chess_2.npy", corners2)
    # Draw and display the corners
    # cv2.drawChessboardCorners(img, (7, 6), corners2, ret)
    # cv2.imshow('img', img)
    # cv2.imwrite("/home/user/code/zed-sdk/mahdi/log/debug_chess_calibration/detections_2.jpeg", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # # ROS dept/camera_info fullHD half truncation
    # cx = 487.22515869140625
    # cy = 258.14642333984375
    # fx = 377.4441223144531
    # fy = 377.4441223144531

    # # ROS dept/camera_info fullHD no truncation
    # cx = 974.4503173828125
    # cy = 516.2928466796875
    # fx = 754.8882446289062
    # fy = 754.8882446289062
    # distort = np.array([])

    # # ROS dept/camera_info HD1200 no truncation
    # cx = 974.64697265625
    # cy = 575.6713256835938
    # fx = 747.6171875
    # fy = 747.6171875
    # distort = np.array([])

    # ROS dept/camera_info HD1200 no truncation(TODO WHY DIFFERENT FROM ABOVE WITH SAME RUN OF ROS ZED?!!!!)
    cx = 945.35302734375
    cy = 624.3286743164062
    fx = 747.6171875
    fy = 747.6171875
    distort = np.array([])

    # # from factory file at /usr/local/zed/settings for [LEFT_CAM_FHD1200]: exactly matching with unrectified left image data at topic /zedxm/zed_node/rgb_raw/camera_info
    # cx = 958.982
    # cy = 598.733
    # fx = 735.453
    # fy = 735.248
    # distort = np.array([0.763826,
    #                     1.1107,
    #                     -0.000101995,
    #                     -0.000179239,
    #                     0.105266,
    #                     0.766447,
    #                     1.14702,
    #                     0.175504])

    # # opencv intirnsic calibration with chessboard
    # cx = 957.28423445
    # cy = 527.75991205
    # fx = 658.94025256
    # fy = 547.48241378
    # distort=np.array([-0.03737721,  0.05280293, -0.00287547, -0.00086759, -0.03238512])

    error_eucledian_all = []
    error_all = []
    rvec_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []
    acceptable_id = {0, 1}  # {2,3,4,5,6,7,10,11,12,14,15,16,17}
    N = acceptable_id.__len__()
    imagePoints_all = np.zeros((54, 2, N))
    O_T_EE_all = np.zeros((4, 4, N))
    j = 0
    for i in acceptable_id:
        cameraMatrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        print("*****i=", i)
        corners2 = np.load(
            "/home/user/code/zed-sdk/mahdi/log/debug_chess_calibration_d/Chessboard_detections_{}.npy".format(str(i)))
        imagePoints_all[:, :, j] = corners2.squeeze()
        O_T_EE_all[:, :, j] = np.load(
            "/home/user/code/zed-sdk/mahdi/log/debug_chess_calibration_d/O_T_EE_{}.npy".format(str(i)))
        color_image = cv2.imread(
            "/home/user/code/zed-sdk/mahdi/log/debug_chess_calibration_d/image_left_{}.jpeg".format(str(i)))
        depth_image = np.load(
            "/home/user/code/zed-sdk/mahdi/log/debug_chess_calibration_d/depth_map_{}.npy".format(str(i)))
        objectPoints = np.zeros((6 * 9, 3), np.float32)
        cube_length = 36.2 / 1000  # [m]
        # # TODO uncomment for eye in hand calibration
        # objectPoints[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * cube_length
        # #TODO uncomment for chessboard in front fixed camera when obj coordinate is the robot coordinate at /home/user/code/zed-sdk/mahdi/log/debug_chess_calibration_d
        tmp = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * np.array([-1, 1])
        objectPoints[:, 0] = tmp[:, 0] * cube_length + 570.5 / 1000
        objectPoints[:, 2] = tmp[:, 1] * cube_length + 20.2 / 1000
        objectPoints[:, 1] += -305 / 1000
        imagePoints = imagePoints_all[:, :, j].reshape((54, 1, 2))
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        objpts = []  # 3d point in real world space
        objpts.append(objectPoints)
        imgpts = []  # 3d point in real world space
        imgpts.append(corners2)
        # ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpts, imgpts, gray.shape[::-1], cameraMatrix,
        #                                                             distort,
        #                                                             flags=cv2.CALIB_USE_INTRINSIC_GUESS)

        # h, w = color_image.shape[:2]
        # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (w, h), 1, (w, h))
        # # undistort
        # dst = cv2.undistort(color_image, cameraMatrix, dist, None, newcameramtx)
        # # crop the image
        # x, y, w, h = roi
        # dst = dst[y:y + h, x:x + w]
        # cv2.imwrite("/home/user/code/zed-sdk/mahdi/log/debug_chess_calibration_d/image_left_{}_calibresult.jpeg".format(str(i)), dst)
        retval, rvecs, tvecs, reprojErr = cv2.solvePnPGeneric(objectPoints, imagePoints, cameraMatrix,
                                                                              distCoeffs=distort,
                                                                              flags=cv2.SOLVEPNP_ITERATIVE)

        H_obj2c = np.vstack((np.hstack((cv2.Rodrigues(rvecs[0])[0], tvecs[0])), np.array([0, 0, 0, 1])))
        for k in range(54):
            x_p_obj_img = imagePoints[k, :, 0]
            y_p_obj_img = imagePoints[k, :, 1]
            u = int(x_p_obj_img)
            v = int(y_p_obj_img)
            Z_depth_map = np.nanmean(depth_image[v - 2: v + 2, u - 2: u + 2])

            P_obj = np.append(objectPoints[k, :], 1)
            p_c_calib = np.matrix(H_obj2c) * np.matrix(P_obj.reshape(4, 1))
            print("k={}\n".format(k))
            print("p_c_calib[2,0]-Z_depth_map={}[mm]".format((p_c_calib[2, 0] - Z_depth_map) * 1000))
