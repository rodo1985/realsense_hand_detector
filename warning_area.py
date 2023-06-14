import os
import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2
from ultralytics import YOLO
import time

def get_bouning_box_pose(depth_frame, center, camera_intrinsics, rect_size=(10, 10)):
    """
    Get the pose of a rectangle in the image with the mean Z value and the world coordinates
    :param depth_frame: depth_frame
    :param center: center of the rectangle
    :param camera_intrinsics: camera intrinsics
    :param rect_size: size of the rectangle
    :return: numpy array with the pose
    """
    
    # get the sizes
    rect_half_height, rect_half_width = rect_size[0] // 2, rect_size[1] // 2
    
    # Clamping the coordinates so the rectangle doesn't go out of bounds
    x_start = int(max(0, center[1] - rect_half_width))
    x_end = int(min(depth_frame.as_depth_frame().width, center[1] + rect_half_width))
    y_start = int(max(0, center[0] - rect_half_height))
    y_end = int(min(depth_frame.as_depth_frame().height, center[0] + rect_half_height))
    
    # iterate over the rectangle
    depth_values = []
    for y in range(y_start, y_end):
        for x in range(x_start, x_end):
            depth_values.append(depth_frame.get_distance(int(x), int(y)))

    #remove 0 values
    depth_values = [x for x in depth_values if x != 0]

    # calculate the mean Z
    Z = np.mean(depth_values)

    # get world coordinates
    X = ( Z * ( int(center[0]) - camera_intrinsics.ppx) / camera_intrinsics.fx)
    Y = Z * (int(center[1])- camera_intrinsics.ppy) / camera_intrinsics.fy

    # numpy array with pose
    return np.array([X, Y, Z])

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
    X = Z * (pixel[0] - camera_intrinsics.ppx) / camera_intrinsics.fx
    Y = Z * (pixel[1] - camera_intrinsics.ppy) / camera_intrinsics.fy

    # numpy array with pose
    return np.array([X, Y, Z])

def process_hand(hand, depth_frame, camera_intrinsics, color_image, RECT_SIZE, top_left_corner, bottom_right_corner):
    """
    Process the hand to get the pose and draw a rectangle on it
    :param hand: hand coordinates
    :param depth_frame: depth_frame
    :param camera_intrinsics: camera intrinsics
    :param color_image: color_image
    :param RECT_SIZE: size of the rectangle
    :param top_left_corner: top left corner of the warning area
    :param bottom_right_corner: bottom right corner of the warning area
    :return: True if the hand is in the warning area
    """

    warning_area = False

    try:
        # get 3D pose
        hand_pose = get_pose(depth_frame, hand, camera_intrinsics)
        print("Hand", hand_pose)

        # check if the hand is in the warning area
        if np.all(hand_pose >= top_left_corner) and np.all(hand_pose <= bottom_right_corner) and hand_pose[2] > 0.0:
            warning_area = True
        
        # draw rectangle on the hands
        cv2.rectangle(color_image, (int(hand[0]-RECT_SIZE[0]/2), int(hand[1]-RECT_SIZE[0]/2)), (int(hand[0]+RECT_SIZE[1]/2), int(hand[1]+RECT_SIZE[1]/2)), (0, 255, 0), 2)
        
    except Exception:
        pass

    return warning_area


def main():

    # contstants
    RECT_SIZE = (20, 20)
    WARNING_BOUNDING_BOX_CENTER = np.array([-0.3, 0.3, 2.2])
    WARNING_BOUNDING_BOX_SIZE = np.array([0.5, 0.5, 0.5])

    # get the min and max corners of the bounding box
    top_left_corner = WARNING_BOUNDING_BOX_CENTER - WARNING_BOUNDING_BOX_SIZE / 2
    bottom_right_corner = WARNING_BOUNDING_BOX_CENTER + WARNING_BOUNDING_BOX_SIZE / 2

    # Load the model
    model = YOLO('yolov8n-pose.pt')

    # Create a pipeline
    pipeline = rs.pipeline()

    # Create a config and configure the pipeline to stream
    config = rs.config()
    config.enable_stream(rs.stream.depth, 320, 240, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # discard first 10 images to wait autoexposure ok
    for _ in range(10):
        frames = pipeline.wait_for_frames()

    # pointcloud object
    pc = rs.pointcloud()

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Streaming loop
    while True:

        # reset warning area
        warning_area = False

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
        original_image = color_image.copy()

        # Predict with the model
        results = model(color_image)

        # check hands are not inside the warning area
        warning_area_left = process_hand(results[0].keypoints.xy[0][9].cpu().numpy(), depth_frame, camera_intrinsics, color_image, RECT_SIZE, top_left_corner, bottom_right_corner)
        warning_area_right = process_hand(results[0].keypoints.xy[0][10].cpu().numpy(), depth_frame, camera_intrinsics, color_image, RECT_SIZE, top_left_corner, bottom_right_corner)

        # set the result
        warning_area = warning_area_left or warning_area_right
        
        # plot the results
        color_image = results[0].plot()

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # get shapes
        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))
    
        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        key = cv2.waitKey(1)

        # press 'q' or 'esc' to exit
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
        # press 's' to save images
        elif key == ord('s'):

            # if folder not exist, create it
            if not os.path.exists('images'):
                os.makedirs('images')

            # get time
            timestamp = frames.get_timestamp()
            
            # save images with time stamp
            cv2.imwrite('images/{}_color.png'.format(timestamp), original_image)
            cv2.imwrite('images/{}_depth.png'.format(timestamp), depth_image)
            
        # press 'c' to capture pointcloud
        elif key == ord('c') or warning_area:

            # calculate points
            points = pc.calculate(depth_frame)

            # convert points to xyz array
            xyz = np.asanyarray(points.get_vertices()).view(np.float32).reshape(
                camera_intrinsics.height * camera_intrinsics.width, 3)

            # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)

            # create the bounding box to cropped the pointcloud
            bbox = o3d.geometry.AxisAlignedBoundingBox(
                min_bound=top_left_corner, max_bound= bottom_right_corner)
            bbox_line_set = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(bbox)
            bbox_line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(bbox_line_set.lines))])  # set color to red
            
            # visualize
            o3d.visualization.draw_geometries(
                            [pcd,bbox_line_set],
                            zoom=0.2,
                            front=[-0.069559414540394313, 0.071052676612831794, -0.99504422263281855],
                            lookat=[0.12577850755419884, 0.091076183622316495, 1.0060000419616699],
                            up=[ -0.012545046355109401, -0.99744369037539493, -0.070347042171349905], 
                            width=960, 
                            height=540)
            
if __name__ == "__main__":
    main()