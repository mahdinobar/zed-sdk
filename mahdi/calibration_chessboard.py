import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cv2

if __name__ == '__main__':
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    fname = '/home/user/code/zed-sdk/mahdi/log/debug_chess_calibration/left_image_3.jpeg'
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)
    # If found, add object points, image points (after refining them)
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    np.save("/home/user/code/zed-sdk/mahdi/log/debug_chess_calibration/corners_chess_2.npy", corners2)
    # Draw and display the corners
    cv2.drawChessboardCorners(img, (7, 6), corners2, ret)
    cv2.imshow('img', img)
    cv2.imwrite("/home/user/code/zed-sdk/mahdi/log/debug_chess_calibration/detections_2.jpeg", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # # ROS dept/camera_info fullHD half truncation
    # cx = 487.22515869140625
    # cy = 258.14642333984375
    # fx = 377.4441223144531
    # fy = 377.4441223144531

    # ROS dept/camera_info fullHD no truncation
    cx = 974.4503173828125
    cy = 516.2928466796875
    fx = 754.8882446289062
    fy = 754.8882446289062
    distort=np.array([])

    # # opencv intirnsic calibration with chessboard
    # cx = 957.28423445
    # cy = 527.75991205
    # fx = 658.94025256
    # fy = 547.48241378
    # distort=np.array([-0.03737721,  0.05280293, -0.00287547, -0.00086759, -0.03238512])

    cameraMatrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    N = 1
    imagePoints_all = np.zeros((42, 2, N))
    O_T_EE_all = np.zeros((4, 4, N))
    for i in range(0, N):
        imagePoints_all[:, :, i] = corners2.squeeze()
        O_T_EE_all[:, :, i] = np.load(
            "/home/user/code/zed-sdk/mahdi/log/debug_chess_calibration/O_T_EE_{}.npy".format(str(i + 1)))
        # objectPoints = np.array([[-0.018, 0.018, 0], [0.018, 0.018, 0], [0.018, -0.018, 0], [-0.018, -0.018, 0]])
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objectPoints = np.zeros((6 * 7, 3), np.float32)
        cube_length=26.8/1000 #[m]
        objectPoints[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)*cube_length

        if i == 0:
            R_gripper2base = O_T_EE_all[:3, :3, i]
            rvec_gripper2base, _ = cv2.Rodrigues(R_gripper2base)
            t_gripper2base = O_T_EE_all[:3, 3, i].reshape((3, 1))
            imagePoints = imagePoints_all[:, :, i].reshape((42, 1, 2))
            retval, R_target2cam, t_target2cam, reprojErr = cv2.solvePnPGeneric(objectPoints, imagePoints, cameraMatrix,
                                                                                distCoeffs=distort,
                                                                                flags=cv2.SOLVEPNP_ITERATIVE)
            R_target2cam = R_target2cam[0]
            t_target2cam = t_target2cam[0]
            print("reprojErr=", reprojErr)

            # objpts = []  # 3d point in real world space
            # objpts.append(objectPoints)
            # imgpts = []  # 3d point in real world space
            # imgpts.append(corners2)
            # ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpts, imgpts, gray.shape[::-1], None, None)
            # R_target2cam=rvecs[0]
            # t_target2cam=tvecs[0]

            for k in range(0,42):
                x_p_obj_img = imagePoints[k,:,0]
                y_p_obj_img = imagePoints[k,:,1]
                u = int(x_p_obj_img)
                v = int(y_p_obj_img)

                # ROS point cloud based
                point_cloud = np.load("/home/user/code/zed-sdk/mahdi/log/debug_chess_calibration/point_cloud_{}.npy".format(str(i + 1)))
                p_c_ROS = point_cloud[v, u, :3]
                p_c=np.zeros(3)
                p_c[0]=-p_c_ROS[1]
                p_c[1]=-p_c_ROS[2]
                p_c[2]=p_c_ROS[0]

                # # intrinsic based
                # depth_image=np.load("/home/user/code/zed-sdk/mahdi/log/debug_chess_calibration/depth_map_2.npy")
                # Z0 = np.nanmean(depth_image[v - 2: v + 2, u - 2: u + 2])
                # X0 = Z0 * (u - cx) / fx
                # Y0 = Z0 * (v - cy) / fy
                # p_c = np.array([X0, Y0, Z0])

                H_t2c = np.vstack((np.hstack((cv2.Rodrigues(R_target2cam[:,-1])[0], t_target2cam[:,-1].reshape(3,1))), np.array([0, 0, 0, 1])))
                P_t = np.append(objectPoints[k, :], 1)
                p_c_calib=np.matrix(H_t2c)*np.matrix(P_t.reshape(4,1))
                error=np.array(p_c_calib)[:3].squeeze()-p_c
                print("k=",k)
                print("error_{}=".format(str(i)),error*1000,"[mm]")
                print("eucledian overall error_{}=".format(str(i)),np.linalg.norm(error)*1000,"[mm]")