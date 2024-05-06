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
    init_params.depth_minimum_distance = 120  # Set the minimum depth perception distance
    init_params.depth_maximum_distance = 220  # Set the maximum depth perception distance
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
    mirror_ref = sl.Transform()
    mirror_ref.set_translation(sl.Translation(2.75, 4.0, 0))
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
            cv2_imshow(image.get_data(), window_name="left image")
            gray_image = cv2.cvtColor(image.get_data(), cv2.COLOR_BGR2GRAY)
            tags = at_detector.detect(gray_image, False, camera_params=None)
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            delta_p = np.array([])
            point_cloud_value = np.zeros((2,3))
            # delta_p_buffer=np.array([])
            tag_idx=0
            for tag in tags:
                for idx in range(len(tag.corners)):
                    print("corner detected with measured relative depth = {0:0.4f} [mm].".format(
                        depth.numpy().T[tag.corners[idx, 0].astype(int), tag.corners[idx, 1].astype(int)]))
                delta_p = np.append(delta_p, depth.numpy().T[
                    np.mean(tag.corners, 0)[0].astype(int), np.mean(tag.corners, 0)[1].astype(int)])
                print("estimated depth for the center of tag = {0:0.4f} [mm].".format(delta_p[-1]))
                # Retrieve colored point cloud. Point cloud is aligned on the left image.
                zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
                # Get and print distance value in mm at the center of the image
                # We measure the distance camera - object using Euclidean distance
                x = round(np.mean(tag.corners, 0)[0])
                y = round(np.mean(tag.corners, 0)[1])
                err, value = point_cloud.get_value(x, y)
                point_cloud_value[tag_idx,:] = value[0:3]
                print("value = {} [mm].".format(value))
                print("err = {}.".format(err))
                tag_idx+=1
            # delta_p_buffer=np.vstack((delta_p_buffer,delta_p))
            i += 1
            zed.close()
            print("point_cloud_value = {} [mm].".format(point_cloud_value))
            delta_p_camera_ee=point_cloud_value[1,:]- point_cloud_value[0,:]
            # we print the estimated relative xyz position of center of april tag square on cube of the conveyor belt
            # WITH RESPECT TO estimated center of april tag xyz position for the cube gripped by the robot had (as
            # estimation of the End Effector(ee) position)
            print("+++++delta_p_ee_cube = {} [mm].".format(delta_p_camera_ee))

if __name__ == "__main__":
    main()
