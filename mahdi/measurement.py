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
    init_params.depth_minimum_distance = 100  # Set the minimum depth perception distance
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

    while i < 1:
        while i < 1:
            # A new image is available if grab() returns SUCCESS
            if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                # Retrieve left image
                zed.retrieve_image(image, sl.VIEW.LEFT)
                cv2_imshow(image.get_data(), window_name="left image")
                cv2.imwrite("/home/user/code/zed-sdk/mahdi/log/testl.jpg", image.get_data())
                # detector = apriltag("tag36h11")
                # detections = detector.detect(image)
                zed.retrieve_image(image, sl.VIEW.RIGHT)
                cv2_imshow(image.get_data(), window_name="right image")
                cv2.imwrite("/home/user/code/zed-sdk/mahdi/log/testr.jpg", image.get_data())
                # Retrieve depth map. Depth is aligned on the left image
                zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
                cv2_imshow(depth.get_data(), window_name="depth map")
                cv2.imwrite("/home/user/code/zed-sdk/mahdi/log/depth.jpg", depth.get_data())
                np.save("/home/user/code/zed-sdk/mahdi/log/depth.npy", depth.get_data())
                # Retrieve colored point cloud. Point cloud is aligned on the left image.
                zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
                # Get and print distance value in mm at the center of the image
                # We measure the distance camera - object using Euclidean distance
                x = round(image.get_width() / 2)
                y = round(image.get_height() / 2)
                err, point_cloud_value = point_cloud.get_value(x, y)

                if math.isfinite(point_cloud_value[2]):
                    distance = math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
                                         point_cloud_value[1] * point_cloud_value[1] +
                                         point_cloud_value[2] * point_cloud_value[2])
                    print(f"Distance to Camera at {{{x};{y}}}: {distance}")
                else:
                    print(f"The distance can not be computed at {{{x};{y}}}")
                i += 1
                # Close the camera
        zed.close()


def detection():
    img = cv2.imread("/home/user/code/zed-sdk/mahdi/log/testl.jpg", cv2.IMREAD_GRAYSCALE)
    img_depth = cv2.imread("/home/user/code/zed-sdk/mahdi/log/depth.jpg", cv2.IMREAD_GRAYSCALE)
    depth_data = np.load("/home/user/code/zed-sdk/mahdi/log/depth.npy")

    at_detector = Detector(searchpath=['apriltags'],
                           families='tag36h11',
                           nthreads=1,
                           quad_decimate=1.0,
                           quad_sigma=0.0,
                           refine_edges=1,
                           decode_sharpening=0.25,
                           debug=0)
    tags = at_detector.detect(img, False, camera_params=None, )
    print(tags)
    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for tag in tags:
        for idx in range(len(tag.corners)):
            cv2.line(color_img, tuple(tag.corners[idx - 1, :].astype(int)), tuple(tag.corners[idx, :].astype(int)),
                     (0, 255, 0))
            cv2.drawMarker(color_img, tuple(tag.corners[idx, :].astype(int)), color=(255, 0, 0))
            cv2.line(img_depth, tuple(tag.corners[idx - 1, :].astype(int)), tuple(tag.corners[idx, :].astype(int)),
                     (0, 255, 0))
            cv2.drawMarker(img_depth, tuple(tag.corners[idx, :].astype(int)), color=(255, 0, 0))
            print("corner detected on image plane location = ({0:0.3f},{0:0.3f}) [pxls] with measured depth = {0:0.3f} [mm].".format(
                tag.corners[idx, 0], tag.corners[idx, 1],
                depth_data.T[tag.corners[idx, 0].astype(int), tag.corners[idx, 1].astype(int)]))

        cv2.putText(color_img, str(tag.tag_id),
                    org=(tag.corners[0, 0].astype(int) + 10, tag.corners[0, 1].astype(int) + 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.8,
                    color=(0, 0, 255))
    cv2.imshow('Detected tags', color_img)
    k = cv2.waitKey(0)
    if k == 27:  # wait for ESC key to exit
        cv2.destroyAllWindows()
    cv2.imwrite("/home/user/code/zed-sdk/mahdi/log/detection.jpg", color_img)
    cv2.imwrite("/home/user/code/zed-sdk/mahdi/log/detection_marked_depth.jpg", img_depth)


if __name__ == "__main__":
    # main()
    detection()
