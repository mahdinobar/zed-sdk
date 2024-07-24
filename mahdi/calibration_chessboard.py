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

    # ROS dept/camera_info HD1200 no truncation
    cx = 974.64697265625
    cy = 575.6713256835938
    fx = 747.6171875
    fy = 747.6171875
    distort = np.array([])

    # # opencv intirnsic calibration with chessboard
    # cx = 957.28423445
    # cy = 527.75991205
    # fx = 658.94025256
    # fy = 547.48241378
    # distort=np.array([-0.03737721,  0.05280293, -0.00287547, -0.00086759, -0.03238512])

    cameraMatrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    N = 10
    imagePoints_all = np.zeros((54, 2, N))
    O_T_EE_all = np.zeros((4, 4, N))
    error_eucledian_all = []
    error_all = []
    rvec_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []
    for i in range(0, N):
        print("*****i=", i)
        corners2 = np.load(
            "/home/user/code/zed-sdk/mahdi/log/debug_chess_calibration_c/Chessboard_detections_{}.npy".format(str(i)))
        imagePoints_all[:, :, i] = corners2.squeeze()
        O_T_EE_all[:, :, i] = np.load(
            "/home/user/code/zed-sdk/mahdi/log/debug_chess_calibration_c/O_T_EE_{}.npy".format(str(i)))

        # objectPoints = np.array([[-0.018, 0.018, 0], [0.018, 0.018, 0], [0.018, -0.018, 0], [-0.018, -0.018, 0]])
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objectPoints = np.zeros((6 * 9, 3), np.float32)
        cube_length = 36.2 / 1000  # [m]
        objectPoints[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * cube_length
        imagePoints = imagePoints_all[:, :, i].reshape((54, 1, 2))
        retval, R_target2cam_, t_target2cam_, reprojErr = cv2.solvePnPGeneric(objectPoints, imagePoints, cameraMatrix,
                                                                              distCoeffs=distort,
                                                                              flags=cv2.SOLVEPNP_ITERATIVE)
        R_target2cam_ = R_target2cam_[0]
        t_target2cam_ = t_target2cam_[0]
        R_target2cam.append(R_target2cam_)
        t_target2cam.append(t_target2cam_)
        print("reprojErr=", reprojErr)

        # img=cv2.imread("/home/user/code/zed-sdk/mahdi/log/debug_chess_calibration_c/image_left_{}.jpeg".format(i))
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # # ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPoints, imagePoints, gray.shape[::-1], None, None)
        # objpts = []  # 3d point in real world space
        # objpts.append(objectPoints)
        # imgpts = []  # 3d point in real world space
        # imgpts.append(corners2)
        # ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpts, imgpts, gray.shape[::-1], None, None)
        # R_target2cam=rvecs[0]
        # t_target2cam=tvecs[0]
        # cx = mtx[0,2]
        # cy = mtx[1,2]
        # fx = mtx[0,0]
        # fy = mtx[1,1]

        # validate_pnp
        for k in range(0,54):
            x_p_obj_img = imagePoints[k,:,0]
            y_p_obj_img = imagePoints[k,:,1]
            u = int(x_p_obj_img)
            v = int(y_p_obj_img)
            # ROS point cloud based
            # point_cloud = np.load("/home/user/code/zed-sdk/mahdi/log/debug_chess_calibration/point_cloud_{}.npy".format(str(i + 1)))
            point_cloud= np.load("/home/user/code/zed-sdk/mahdi/log/debug_chess_calibration_c/ROS_point_cloud_{}.npy".format(str(i)))
            p_c_ROS = point_cloud[v, u, :3]
            p_c=np.zeros(3)
            p_c[0]=-p_c_ROS[1]
            p_c[1]=-p_c_ROS[2]
            p_c[2]=p_c_ROS[0]
            # # intrinsic based
            # depth_image=np.load("/home/user/code/zed-sdk/mahdi/log/debug_chess_calibration_c/depth_map_{}.npy".format(str(i)))
            # Z0 = np.nanmean(depth_image[v - 2: v + 2, u - 2: u + 2])
            # X0 = Z0 * (u - cx) / fx
            # Y0 = Z0 * (v - cy) / fy
            # p_c = np.array([X0, Y0, Z0])

            H_t2c = np.vstack((np.hstack((cv2.Rodrigues(R_target2cam_[:,-1])[0], t_target2cam_[:,-1].reshape(3,1))), np.array([0, 0, 0, 1])))
            P_t = np.append(objectPoints[k, :], 1)
            p_c_calib=np.matrix(H_t2c)*np.matrix(P_t.reshape(4,1))
            error=np.array(p_c_calib)[:3].squeeze()-p_c#-np.array([0,0,0.005654180201442943])
            print("k=",k)
            print("error_{}=".format(str(i)),error*1000,"[mm]")
            print("eucledian overall error_{}=".format(str(i)),np.linalg.norm(error)*1000,"[mm]")
            error_eucledian_all.append(np.linalg.norm(error) * 1000)
            error_all.append(error)


        rvec_gripper2base.append(cv2.Rodrigues(O_T_EE_all[:3, :3, i])[0])
        t_gripper2base.append(O_T_EE_all[:3, 3, i].reshape((3, 1)))
    error_all=np.array(error_all).reshape(10, 54, 3)
    error_eucledian_all=np.array(error_eucledian_all).reshape(10, 54)
    print("nan mean overall error per image:",np.nanmean(error_eucledian_all,1))
    np.save("/home/user/code/zed-sdk/mahdi/log/debug_chess_calibration_c/error_PnP.npy",error_all)
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(np.array(rvec_gripper2base), np.array(t_gripper2base),
                                                        np.array(R_target2cam),
                                                        np.asarray(t_target2cam),
                                                        method=cv2.CALIB_HAND_EYE_TSAI)
    print("R_cam2gripper=", R_cam2gripper)
    print("t_cam2gripper=", t_cam2gripper)

    t_gripper2cam = -np.matrix(R_cam2gripper).T * np.matrix(t_cam2gripper)
    R_gripper2cam = np.matrix(R_cam2gripper).T
    print("R_gripper2cam=", R_gripper2cam)
    print("t_gripper2cam=", t_gripper2cam)

    np.save("/home/user/code/zed-sdk/mahdi/log/debug_chess_calibration_c/R_cam2gripper.npy", R_cam2gripper)
    np.save("/home/user/code/zed-sdk/mahdi/log/debug_chess_calibration_c/t_cam2gripper.npy", t_cam2gripper)
