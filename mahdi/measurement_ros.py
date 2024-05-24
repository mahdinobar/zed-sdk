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
    init_params.depth_minimum_distance = 150  # Set the minimum depth perception distance
    init_params.depth_maximum_distance = 450  # Set the maximum depth perception distance
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
                        "!!corner detected on image plane location = ({},{}) [pxls] with measured depth map value= {} [mm].".format(
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
                if tag_idx==0: #TODO make this and else conditions robust to order of tags
                    # retrieve sensors data
                    zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.IMAGE)
                    # # Extract IMU data
                    # imu_data = sensors_data.get_imu_data()
                    # print("imu_data.get_pose().m = ", imu_data.get_pose().m)
                    # Retrieve colored point cloud. Point cloud is aligned on the left image.
                    zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
                    # Get and print distance value in mm at the center of the image
                    # We measure the position of the mid point of the lower edge of the apriltag
                    x_t_ftc2_img, y_t_ftc2_img =(tag.corners[0] + tag.corners[2]) / 2
                    err_t_ftc2_ca, t_ftc2_ca = point_cloud.get_value(int(x_t_ftc2_img), int(y_t_ftc2_img))
                    t_ftc2_ca=t_ftc2_ca[0: 3]
                    # point_cloud_value[tag_idx, :] = t_ftc2_ca[0:3]
                    print("t_ftc2_ca = {} [mm].".format(t_ftc2_ca))
                    print("err_t_ftc2_ca = {}.".format(err_t_ftc2_ca))

                    x_y_ftc2_img, y_y_ftc2_img = tag.corners[2]
                    err_y_ftc2_ca, y_ftc2_ca = point_cloud.get_value(int(x_y_ftc2_img), int(y_y_ftc2_img))
                    y_ftc2_ca=y_ftc2_ca[0: 3]
                    print("y_ftc2_ca = {} [mm].".format(y_ftc2_ca))
                    print("err_y_ftc2_ca = {}.".format(err_y_ftc2_ca))

                    x_c_tag_img, y_c_tag_img = np.mean(tag.corners,0)
                    err_c_tag_ca, c_tag_ca = point_cloud.get_value(int(x_c_tag_img), int(y_c_tag_img))
                    c_tag_ca=c_tag_ca[0: 3]
                    print("y_c_tag_img = {} [mm].".format(y_c_tag_img))
                    print("err_c_tag_ca = {}.".format(err_c_tag_ca))

                    z_ftc2_ca = t_ftc2_ca + (t_ftc2_ca - c_tag_ca)

                    x_ftc2_ca = np.cross(y_ftc2_ca, z_ftc2_ca)

                    R=np.vstack((x_ftc2_ca - t_ftc2_ca, y_ftc2_ca - t_ftc2_ca, z_ftc2_ca - t_ftc2_ca))
                    T_ca_ftc2=np.vstack((np.hstack((R, t_ftc2_ca.reshape(3, 1))), np.array([0, 0, 0, 1])))
                elif tag_idx==1:
                    zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.IMAGE)
                    zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
                    # get position of edge of apriltag in the corner of robot table
                    x_p_obj_img, y_p_obj_img = tag.corners[2]
                    err_p_obj_ca, p_obj_ca = point_cloud.get_value(int(x_p_obj_img), int(y_p_obj_img))
                    p_obj_ca=p_obj_ca[0: 3]
                    print("p_obj_ca = {} [mm].".format(p_obj_ca))
                    print("err_p_obj_ca = {}.".format(err_p_obj_ca))
                tag_idx += 1

            i += 1
            cv2_imshow(color_image, window_name="left image")
            cv2_imshow(depth.get_data(), window_name="depth map")

    zed.close()
    return(p_obj_ca, T_ca_ftc2)


if __name__ == "__main__":
    p_obj_ca, T_ca_ftc2 = main()
