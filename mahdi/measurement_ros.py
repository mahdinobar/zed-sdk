########################################################################
#
# Copyright (c) 2024, Mahdi Nobar.
#
# All rights reserved.
#
########################################################################
import pyzed.sl as sl
import math
import numpy as np
import sys
import math
import cv2
from dt_apriltags import Detector


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


def main():
    # Create a Camera object
    zed = sl.Camera()
    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL  # Use ULTRA depth mode
    init_params.coordinate_units = sl.UNIT.MILLIMETER  # Use meter units (for depth measurements)
    init_params.depth_minimum_distance = 110  # Set the minimum depth perception distance
    init_params.depth_maximum_distance = 250  # Set the maximum depth perception distance
    # Open the camera
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:  # Ensure the camera has opened succesfully
        print("Camera Open : " + repr(status) + ". Exit program.")
        exit()
    # Create and set RuntimeParameters after opening the camera
    runtime_parameters = sl.RuntimeParameters()
    i = 0
    image = sl.Mat()
    depth = sl.Mat()
    point_cloud = sl.Mat()
    sensors_data = sl.SensorsData()
    at_detector = Detector(searchpath=['apriltags'],
                           families='tag36h11',
                           nthreads=1,
                           quad_decimate=1.0,
                           quad_sigma=0.0,
                           refine_edges=1,
                           decode_sharpening=0.25,
                           debug=0)
    while i < 1:
        print(">>>Frame number = {:d}".format(i))
        # A new image is available if grab() returns SUCCESS
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image
            zed.retrieve_image(image, sl.VIEW.LEFT)
            color_image=image.get_data()
            # cv2_imshow(color_image, window_name="left image")
            gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            # gray_image = cv2.imread("/home/user/code/zed-sdk/mahdi/log/testl_2.jpg", cv2.IMREAD_GRAYSCALE)
            # img_depth = cv2.imread("/home/user/code/zed-sdk/mahdi/log/depth_2.jpg", cv2.IMREAD_GRAYSCALE)
            # depth = np.load("/home/user/code/zed-sdk/mahdi/log/depth_2.npy")
            tags = at_detector.detect(gray_image, False, camera_params=None)
            delta_p = np.array([])
            point_cloud_value = np.zeros((2, 3))
            # delta_p_buffer=np.array([])
            tag_idx = 0
            for tag in tags:
                for idx in range(len(tag.corners)):
                    # print("corner detected with measured relative depth = {0:0.3f} [mm].".format(
                    #     depth.T[tag.corners[idx, 0].astype(int), tag.corners[idx, 1].astype(int)]))
                    print(
                        "!!corner detected on image plane location = ({0:0.3f},{0:0.3f}) [pxls] with measured depth map value= {0:0.3f}.".format(
                            tag.corners[idx, 0], tag.corners[idx, 1],
                            depth.numpy().T[tag.corners[idx, 0].astype(int), tag.corners[idx, 1].astype(int)]))
                    cv2.line(color_image, tuple(tag.corners[idx - 1, :].astype(int)),
                             tuple(tag.corners[idx, :].astype(int)),
                             (0, 255, 0))
                    cv2.drawMarker(color_image, tuple(tag.corners[idx, :].astype(int)), color=(255, 0, 0))
                    cv2.putText(color_image, str(idx),
                                org=(tag.corners[idx, 0].astype(int) + 3, tag.corners[idx, 1].astype(int) + 3),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=1,
                                color=(255, 0, 0))
                    cv2.line(depth.get_data(), tuple(tag.corners[idx - 1, :].astype(int)),
                             tuple(tag.corners[idx, :].astype(int)),
                             (0, 255, 0))
                    cv2.drawMarker(depth.get_data(), tuple(tag.corners[idx, :].astype(int)), color=(255, 0, 0))
                delta_p = np.append(delta_p, depth.numpy().T[
                    np.mean(tag.corners, 0)[0].astype(int), np.mean(tag.corners, 0)[1].astype(int)])
                print("estimated depth for the center of tag = {0:0.4f} [mm].".format(delta_p[-1]))
                # # retrieve sensors data
                # zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.IMAGE)
                # # Extract IMU data
                # imu_data = sensors_data.get_imu_data()
                # print("imu_data.get_pose().m = ", imu_data.get_pose().m)
                # # Retrieve colored point cloud. Point cloud is aligned on the left image.
                # zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
                # # Get and print distance value in mm at the center of the image
                # # We measure the distance camera - object using Euclidean distance
                # x = round(np.mean(tag.corners, 0)[0])
                # y = round(np.mean(tag.corners, 0)[1])
                # err, value = point_cloud.get_value(x, y)
                # point_cloud_value[tag_idx, :] = value[0:3]
                # print("value = {} [mm].".format(value))
                # print("err = {}.".format(err))
                # # estimate pixels of center of ee coordinate system at midpoint of the lower edge of the april tag on
                # # the cube gripped by robot hand
                # o_c_ee_uv = (tag.corners[0, :] + tag.corners[1, :]) / 2
                # err, o_ee_c = point_cloud.get_value(round(o_c_ee_uv[0]), round(o_c_ee_uv[1]))
                # err, ox = point_cloud.get_value(round(tag.corners[1, 0]), round(tag.corners[1, 1]))
                # ox_ee_c = ox - o_ee_c
                # err, oy = point_cloud.get_value(round(tag.corners[3, 0]), round(tag.corners[3, 1]))
                # oy_ee_c = oy - o_ee_c
                # oz_ee_c = np.cross(ox_ee_c[0:3], oy_ee_c[0:3])
                # o_ee_c = o_ee_c[0:3]
                # ox_ee_c = ox_ee_c[0:3] / np.linalg.norm(ox_ee_c[0:3])
                # oy_ee_c = oy_ee_c[0:3] / np.linalg.norm(oy_ee_c[0:3])
                # oz_ee_c = oz_ee_c[0:3] / np.linalg.norm(oz_ee_c[0:3])
                # R = np.vstack((ox_ee_c - o_ee_c, oy_ee_c - o_ee_c, oz_ee_c - o_ee_c))
                # T_c_ee = np.vstack((np.vstack((R, o_ee_c)).T,
                #                     [0, 0, 0, 1]))  # [ox_c_c,oy_c_c,oz_c_c]R+o_ee_c=[ox_ee_c,oy_ee_c,oz_ee_c]
                # print("++++++o_ee_c = ", o_ee_c)
                # print("++++++T_c_ee = ", T_c_ee)

                tag_idx += 1
            # delta_p_buffer=np.vstack((delta_p_buffer,delta_p))
            # print("point_cloud_value = {} [mm].".format(point_cloud_value))
            # delta_p_ee_object = point_cloud_value[1, :] - point_cloud_value[0, :]
            # # we print the estimated relative xyz position of center of april tag square on cube of the conveyor belt
            # # WITH RESPECT TO estimated center of april tag xyz position for the cube gripped by the robot had (as
            # # estimation of the End Effector(ee) position)
            # if delta_p_ee_object[2] < 0:
            #     delta_p_ee_object = -delta_p_ee_object
            # print("delta_p_ee_object = {} [mm].".format(delta_p_ee_object))
            i += 1
            cv2_imshow(color_image, window_name="left image")
            cv2_imshow(depth.get_data(), window_name="depth map")

    zed.close()


if __name__ == "__main__":
    main()
