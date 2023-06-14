import os
import pyrealsense2 as rs
import numpy as np
import cv2
import time

def get_pose(depth_frame, pixel, camera_intrinsics):
    """
    Get the pose of a pixel in the image with the Z value and the world coordinates
    :param depth_frame: depth_frame
    :param pixel: pixel coordinates
    :param camera_intrinsics: camera intrinsics
    :return: numpy array with the pose
    """
    
    # get depth value
    Z = depth_frame.get_distance(int(pixel[0]), int(pixel[1]))

    # get world coordinates
    X = (Z * (pixel[0] - camera_intrinsics.ppx) / camera_intrinsics.fx)
    Y = Z * (pixel[1] - camera_intrinsics.ppy) / camera_intrinsics.fy

    # numpy array with pose
    return np.array([X, Y, Z])


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pose = get_pose(param["depth_frame"], (x, y), param["camera_intrinsics"])
        print(f"Pose: {pose}")


def main():

    # Create a pipeline
    pipeline = rs.pipeline()

    # Create a config and configure the pipeline to stream
    config = rs.config()
    config.enable_stream(rs.stream.depth, 320, 240, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Create an align object
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Streaming loop
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        
        # Align the depth frame to color frame
        aligned_frames = align.process(frames)
        
        # get aligned frames
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # get the intrinsics of the camera
        camera_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback('RealSense', mouse_callback, {"depth_frame": depth_frame, "camera_intrinsics": camera_intrinsics})
        cv2.imshow('RealSense', color_image)
        key = cv2.waitKey(1)

        # press 'q' or 'esc' to exit
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    main()
