import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cv2

if __name__ == '__main__':
    # # 0.0652981  -0.997677 -0.0188917   0.266184
    # # -0.96389 -0.0581667  -0.259834  -0.362177
    # # 0.258132  0.0351762  -0.965468   0.226292
    # # 0          0          0          1
    #
    cx = 487.22515869140625
    cy = 258.14642333984375
    fx = 377.4441223144531
    fy = 377.4441223144531
    cameraMatrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    N = 10
    imagePoints_all = np.zeros((4, 2, N))
    O_T_EE_all = np.zeros((4, 4, N))
    for i in range(0, N):

        imagePoints_all[:, :, i] = np.load(
            "/home/user/code/zed-sdk/mahdi/log/debug_calibration/2Ddetections_{}.npy".format(str(i + 1)))
        O_T_EE_all[:, :, i] = np.load(
            "/home/user/code/zed-sdk/mahdi/log/debug_calibration/O_T_EE_{}.npy".format(str(i + 1)))
        objectPoints = np.array([[-0.018, 0.018, 0], [0.018, 0.018, 0], [0.018, -0.018, 0], [-0.018, -0.018, 0]])

        if i == 0:
            # conf #1
            # R_gripper2base = np.array(
            #     [[0.0672185, -0.997396, -0.0257519], [-0.965466, -0.0585129, -0.253837], [0.251669, 0.0419251, -0.966904]]).T
            # R_gripper2base = O_T_EE_all[:3,:3,i].T
            R_gripper2base = O_T_EE_all[:3, :3, i]
            rvec_gripper2base, _ = cv2.Rodrigues(R_gripper2base)
            # t_gripper2base = np.dot(-R_gripper2base.T, np.array([[0.268861, -0.354426, 0.221448]]).reshape((3, 1)))
            # t_gripper2base = np.dot(-R_gripper2base.T, O_T_EE_all[:3,3,0].reshape((3, 1)))
            # t_gripper2base=-np.matrix(R_gripper2base).T * np.matrix(O_T_EE_all[:3,3,i].reshape((3, 1)))
            t_gripper2base = O_T_EE_all[:3, 3, i].reshape((3, 1))
            # t_gripper2base = np.array([[0.268861, -0.354426, 0.221448]])
            # imagePoints = np.array([[527.19995117, 367.96914673], [507.27575684, 404.05557251], [471.99334717, 381.89071655],
            #                          [493.07861328, 347.8132019]])[:,[1,0]].reshape((4, 1, 2))
            imagePoints = imagePoints_all[:, :, i].reshape((4, 1, 2))

            if True:
                # path
                path = '/home/user/code/zed-sdk/mahdi/log/debug_calibration/left_image_{}.jpeg'.format(str(i + 1))
                # Reading an image in default mode
                image = cv2.imread(path)
                # Window name in which image is displayed
                window_name = 'left rectified image'
                # Using cv2.imshow() method
                # Displaying the image
                cv2.imshow(window_name, image)
                # waits for user to press any key
                # (this is necessary to avoid Python kernel form crashing)
                cv2.waitKey(0)
                # closing all open windows
                cv2.destroyAllWindows()

            retval, R_target2cam, t_target2cam, reprojErr = cv2.solvePnPGeneric(objectPoints, imagePoints, cameraMatrix,
                                                              distCoeffs=np.array([]), flags=cv2.SOLVEPNP_ITERATIVE)
            R_target2cam=R_target2cam[0]
            t_target2cam=t_target2cam[0]
            print("reprojErr=",reprojErr)

            if True:
                # cx = 487.22515869140625
                # cy = 258.14642333984375
                # fx = 377.4441223144531
                # fy = 377.4441223144531
                x_p_obj_img = imagePoints[0,:,0]
                y_p_obj_img = imagePoints[0,:,1]
                u = int(x_p_obj_img)
                v = int(y_p_obj_img)
                # path = '/home/user/code/zed-sdk/mahdi/log/debug_calibration/depth_map_{}.jpeg'.format(str(i + 1))
                # # Reading an image in default mode
                # depth_image = cv2.imread(path)
                # Z0 = np.nanmean(depth_image[v - 2: v + 2, u - 2: u + 2])
                # X0 = Z0 * (u - cx) / fx
                # Y0 = Z0 * (v - cy) / fy
                # p_obj_ca = np.array([X0, Y0, Z0])
                point_cloud = np.load("/home/user/code/zed-sdk/mahdi/log/debug_calibration/point_cloud_{}.npy".format(str(i + 1)))
                p_c_ROS = point_cloud[v, u, :3]
                # ax.scatter(Z0, -X0, -Y0, color="blue")
                # ax.scatter(X, Y, Z, color="red")
                p_c=np.zeros(3)
                p_c[0]=-p_c_ROS[1]
                p_c[1]=-p_c_ROS[2]
                p_c[2]=p_c_ROS[0]

                # true_t_camINtarget_idk = -np.matrix(cv2.Rodrigues(R_target2cam)[0]).T * np.matrix(t_target2cam)
                # prospective_projection = np.vstack((np.vstack((np.eye(3), np.zeros(3))).T,np.zeros(4)))
                H_t2c = np.vstack((np.hstack((cv2.Rodrigues(R_target2cam[:,-1])[0], t_target2cam[:,-1].reshape(3,1))), np.array([0, 0, 0, 1])))
                P_t = np.append(objectPoints[0, :], 1)
                # cameraMatrix2= np.vstack((np.vstack((cameraMatrix, np.zeros(3))).T, np.zeros(4)))
                # I = cameraMatrix2 * prospective_projection * H_t2c * P_t

                p_c_calib=np.matrix(H_t2c)*np.matrix(P_t.reshape(4,1))
                error=np.array(p_c_calib)[:3].squeeze()-p_c
                print("error_{}=".format(str(i)),error*1000,"[mm]")
                print("eucledian overall error_{}=".format(str(i)),np.linalg.norm(error)*1000,"[mm]")
                # fig = plt.figure(1)
                # ax = plt.axes(projection='3d')
                # ax.scatter(point_cloud[:,:,0], point_cloud[:,:,1], point_cloud[:,:,2], color="blue")
                # ax.scatter(X, Y, Z, color="red")

            # t_target2cam=-np.matrix(cv2.Rodrigues(R_target2cam)[0]).T * np.matrix(t_target2cam)
            # R_target2cam=cv2.Rodrigues(np.matrix(cv2.Rodrigues(R_target2cam)[0]).T)[0]
        else:
            # conf #2
            # R_gripper2base = np.array(
            #     [[-0.00239853, -0.999985, -0.00215726], [-0.999974, 0.0023874, 0.00514711],
            #      [-0.00514189, 0.00216955, -0.999984]]).T
            #     R_gripper2base = O_T_EE_all[:3,:3,i].T
            R_gripper2base = O_T_EE_all[:3, :3, i]
            rvec_gripper2base = np.hstack((rvec_gripper2base, cv2.Rodrigues(R_gripper2base)[0]))
            # t_gripper2base = np.hstack((t_gripper2base, -np.matrix(R_gripper2base).T * np.matrix(O_T_EE_all[:3,3,i].reshape((3, 1)))))
            t_gripper2base = np.hstack((t_gripper2base, O_T_EE_all[:3, 3, i].reshape((3, 1))))
            imagePoints = imagePoints_all[:, :, i].reshape((4, 1, 2))
            # imagePoints = np.array([[500.85449219, 352.75317383], [483.55334473, 382.2131958], [450.7286377, 365.84387207],
            #                          [469.37045288, 338.23959351]])[:,[1,0]].reshape((4, 1, 2))
            _, R_target2cam_, t_target2cam_,reprojErr = cv2.solvePnPGeneric(objectPoints, imagePoints, cameraMatrix,
                                                           distCoeffs=np.array([]), flags=cv2.SOLVEPNP_ITERATIVE)
            R_target2cam_=R_target2cam_[0]
            t_target2cam_=t_target2cam_[0]
            print("reprojErr=",reprojErr)
            # t_target2cam_ = -np.matrix(cv2.Rodrigues(R_target2cam_)[0]).T * np.matrix(t_target2cam_)
            # R_target2cam_ = cv2.Rodrigues(np.matrix(cv2.Rodrigues(R_target2cam_)[0]).T)[0]
            R_target2cam = np.hstack((R_target2cam, R_target2cam_))
            t_target2cam = np.hstack((t_target2cam, t_target2cam_))

            if True:
                x_p_obj_img = imagePoints[0,:,0]
                y_p_obj_img = imagePoints[0,:,1]
                u = int(x_p_obj_img)
                v = int(y_p_obj_img)
                point_cloud_ROS = np.load("/home/user/code/zed-sdk/mahdi/log/debug_calibration/point_cloud_{}.npy".format(str(i + 1)))
                p_c_ROS = point_cloud_ROS[v, u, :3]
                # ax.scatter(Z0, -X0, -Y0, color="blue")
                # ax.scatter(X, Y, Z, color="red")
                p_c=np.zeros(3)
                p_c[0]=-p_c_ROS[1]
                p_c[1]=-p_c_ROS[2]
                p_c[2]=p_c_ROS[0]

                # true_t_camINtarget_idk = -np.matrix(cv2.Rodrigues(R_target2cam)[0]).T * np.matrix(t_target2cam)
                # prospective_projection = np.vstack((np.vstack((np.eye(3), np.zeros(3))).T,np.zeros(4)))
                H_t2c = np.vstack((np.hstack((cv2.Rodrigues(R_target2cam[:,-1])[0], t_target2cam[:,-1].reshape(3,1))), np.array([0, 0, 0, 1])))
                P_t = np.append(objectPoints[0, :], 1)
                # cameraMatrix2= np.vstack((np.vstack((cameraMatrix, np.zeros(3))).T, np.zeros(4)))
                # I = cameraMatrix2 * prospective_projection * H_t2c * P_t

                p_c_calib=np.matrix(H_t2c)*np.matrix(P_t.reshape(4,1))
                error=np.array(p_c_calib)[:3].squeeze()-p_c
                print("error_{}=".format(str(i)),error*1000,"[mm]")
                print("eucledian overall error_{}=".format(str(i)),np.linalg.norm(error)*1000,"[mm]")
            if False:
                # validate P_ROS data
                fig = plt.figure(1)
                ax = plt.axes(projection='3d')
                ax.scatter(p_c[0],p_c[1],p_c[2], color="blue",s=300)
                N_=30
                point_cloud=np.zeros((N_*2,N_*2,3))
                point_cloud[:,:,0]=-point_cloud_ROS[v-N_:v+N_,u-N_:u+N_,1]
                point_cloud[:,:,1]=-point_cloud_ROS[v-N_:v+N_,u-N_:u+N_,2]
                point_cloud[:,:,2]=point_cloud_ROS[v-N_:v+N_,u-N_:u+N_,0]
                ax.scatter(point_cloud[:,:,0],point_cloud[:,:,1],point_cloud[:,:,2], color="red")
                ax.set_xlabel('x', fontsize=12)
                ax.set_ylabel('y', fontsize=12)
                ax.set_zlabel('z', fontsize=12)
                plt.show()


    # # conf #3
    # R_gripper2base = np.array(
    #     [[0.0260667, -0.999621, 0.00763459], [-0.902239, -0.0202376, 0.430742], [-0.430425, -0.0181163, -0.902443]]).T
    # rvec_gripper2base = np.hstack((rvec_gripper2base, cv2.Rodrigues(R_gripper2base)[0]))
    # t_gripper2base = np.hstack((t_gripper2base, np.dot(-R_gripper2base.T, np.array([[0.264165, -0.46133, 0.180161]]).reshape((3, 1)))))
    # objectPoints = np.array([[-0.018, 0.018, 0], [0.018, 0.018, 0], [0.018, -0.018, 0], [-0.018, -0.018, 0]])
    # imagePoints = np.array([[509.67565918, 386.98773193], [489.73657227, 411.4519043], [452.18103027, 395.74090576],
    #                          [474.29864502, 373.86239624]])[:,[1,0]].reshape((4, 1, 2))
    # _, R_target2cam_, t_target2cam_ = cv2.solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs=np.array([]), flags=cv2.SOLVEPNP_IPPE_SQUARE)
    # R_target2cam = np.hstack((R_target2cam, R_target2cam_))
    # t_target2cam = np.hstack((t_target2cam, t_target2cam_))
    #
    # # conf #4
    # R_gripper2base = np.array(
    #     [[0.0962972, -0.99506, -0.0237383], [-0.652253, -0.0450714, -0.756655], [0.751847, 0.0883471, -0.653383]]).T
    # rvec_gripper2base = np.hstack((rvec_gripper2base, cv2.Rodrigues(R_gripper2base)[0]))
    # t_gripper2base = np.hstack((t_gripper2base, np.dot(-R_gripper2base.T, np.array([[0.263393, -0.2072, 0.300575]]).reshape((3, 1)))))
    # objectPoints = np.array([[-0.018, 0.018, 0], [0.018, 0.018, 0], [0.018, -0.018, 0], [-0.018, -0.018, 0]])
    # imagePoints = np.array([[528.44854736, 363.60913086], [512.86773682, 388.60891724], [487.01522827, 372.58084106],
    #                          [502.3638916, 347.37054443]])[:,[1,0]].reshape((4, 1, 2))
    # _, R_target2cam_, t_target2cam_ = cv2.solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs=np.array([]), flags=cv2.SOLVEPNP_IPPE_SQUARE)
    # R_target2cam = np.hstack((R_target2cam, R_target2cam_))
    # t_target2cam = np.hstack((t_target2cam, t_target2cam_))
    # # conf #5
    # R_gripper2base = np.array(
    #     [[0.0399165, -0.99664, -0.0713881], [-0.410899, 0.0487508, -0.910374], [0.910796, 0.0656723, -0.407581]]).T
    # rvec_gripper2base = np.hstack((rvec_gripper2base, cv2.Rodrigues(R_gripper2base)[0]))
    # # t_gripper2base = np.vstack((t_gripper2base, np.array([[0.267949, -0.16987, 0.311767]])))
    # t_gripper2base = np.hstack((t_gripper2base, np.dot(-R_gripper2base.T, np.array([[0.267949, -0.16987, 0.311767]]).reshape((3, 1)))))
    # # objectPoints = np.array([[0, 0.036, 0], [0.036, 0.036, 0], [0.036, 0, 0], [0, 0, 0]])
    # objectPoints = np.array([[-0.018, 0.018, 0], [0.018, 0.018, 0], [0.018, -0.018, 0], [-0.018, -0.018, 0]])
    # imagePoints = np.array([[528.24511719, 309.82427979], [515.98455811, 331.05734253], [491.75152588, 320.12359619],
    #                          [503.47674561, 298.36868286]])[:,[1,0]].reshape((4, 1, 2))
    # _, R_target2cam_, t_target2cam_ = cv2.solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs=np.array([]), flags=cv2.SOLVEPNP_IPPE_SQUARE)
    # R_target2cam = np.hstack((R_target2cam, R_target2cam_))
    # t_target2cam = np.hstack((t_target2cam, t_target2cam_))
    # # cv2.calibrateHandEye(np.tile(rvec_gripper2base, 4).T, np.tile(t_gripper2base, 4).reshape(3, 4).T,
    # #                      np.tile(R_target2cam, 4).T, np.tile(t_target2cam, 4).T, method=cv2.CALIB_HAND_EYE_TSAI)
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(rvec_gripper2base.T, np.array(t_gripper2base.T), R_target2cam.T,
                                                        np.asarray(t_target2cam.T),
                                                        method=cv2.CALIB_HAND_EYE_TSAI)
    print("R_cam2gripper=", R_cam2gripper)
    print("t_cam2gripper=", t_cam2gripper)

    t_gripper2cam = -np.matrix(R_cam2gripper).T * np.matrix(t_cam2gripper)
    R_gripper2cam = np.matrix(R_cam2gripper).T
    print("R_gripper2cam=", R_gripper2cam)
    print("t_gripper2cam=", t_gripper2cam)

    np.save("/home/user/code/zed-sdk/mahdi/log/debug_calibration/R_cam2gripper.npy", R_cam2gripper)

# hand_coords = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [
#     1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
#
# eye_coords = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0],
#                        [1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
#
# # rotation matrix between the target and camera
# R_target2cam = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [
#     0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
#
# # translation vector between the target and camera
# t_target2cam = np.array([0.0, 0.0, 0.0, 0.0])
#
# # transformation matrix
# T, _ = cv2.calibrateHandEye(hand_coords, eye_coords,
#                             R_target2cam, t_target2cam)
if False:
    X0 = np.load("/home/user/code/zed-sdk/mahdi/log/X0_f.npy")
    Y0 = np.load("/home/user/code/zed-sdk/mahdi/log/Y0_f.npy")
    Z0 = np.load("/home/user/code/zed-sdk/mahdi/log/Z0_f.npy")
    X = np.load("/home/user/code/zed-sdk/mahdi/log/X_f.npy")
    Y = np.load("/home/user/code/zed-sdk/mahdi/log/Y_f.npy")
    Z = np.load("/home/user/code/zed-sdk/mahdi/log/Z_f.npy")

    fig = plt.figure(1)
    ax = plt.axes(projection='3d')
    ax.scatter(Z0, -X0, -Y0, color="blue")
    ax.scatter(X, Y, Z, color="red")
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_zlabel('z', fontsize=12)
    # #
    fig2 = plt.figure(2)
    ax2 = plt.axes(projection='3d')
    errorY = Y + X0
    errorY = np.nan_to_num(errorY, nan=0)
    errorZ = Z + Y0
    errorZ = np.nan_to_num(errorZ, nan=0)
    # ax2.plot_surface(Y, Z, errorY, color="blue")
    # ax2.plot_surface(Y, Z, errorZ, color="red")
    ax2.scatter(Y, Z, np.abs(errorY), color="blue")
    # ax2.scatter(Y, Z, np.abs(errorY), color="red")
    ax2.contour(Y, Z, np.abs(errorY), zdir="z", offset=0, cmap='coolwarm')
    ax2.set_xlabel('y', fontsize=12)
    ax2.set_ylabel('z', fontsize=12)
    ax2.set_zlabel('error', fontsize=12)
    ax2.view_init(elev=0, azim=90, roll=0)

    fig3 = plt.figure(3)
    ax3 = plt.axes(projection='3d')
    ax3.scatter(Y, Z, np.abs(errorZ), color="red")
    ax3.contour(Y, Z, np.abs(errorZ), zdir="z", offset=0, cmap='coolwarm')
    ax3.set_xlabel('y', fontsize=12)
    ax3.set_ylabel('z', fontsize=12)
    ax3.set_zlabel('error', fontsize=12)
    ax3.view_init(elev=0, azim=90, roll=0)
    plt.show()

    fig4 = plt.figure(4)
    plt.imshow(np.abs(np.nan_to_num(errorZ, nan=0)), cmap='Greys', interpolation='nearest',
               norm=LogNorm(vmin=1e-6, vmax=1e-5))
    plt.colorbar()
    plt.show()

    fig5 = plt.figure(5)
    plt.imshow(np.abs(np.nan_to_num(errorY, nan=0)), cmap='Greys', interpolation='nearest',
               norm=LogNorm(vmin=1e-6, vmax=1e-5))
    plt.colorbar()
    plt.show()
