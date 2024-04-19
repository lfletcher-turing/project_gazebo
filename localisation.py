#!/bin/python3

"""
localisation.py
"""

from typing import List
import rclpy
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from vision_msgs.msg import Detection3DArray, Detection3D
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
import numpy as np
from geometry_msgs.msg import PointStamped
from yolov8_msgs.msg import DetectionArray
import numpy as np
import tf2_geometry_msgs
from builtin_interfaces.msg import Duration


class RelativeLocalizer:
    """A relative localization wrapper that precomputes the inverse camera matrix"""

    def __init__(self, K, D):
        self.K = K
        self.D = D
        self.K_inv = np.linalg.inv(K)  # precompute for efficiency!

    def detection_to_bearing(self, bbox_center, bbox_size, object_size):
        K, D, K_inv = self.K, self.D, self.K_inv
        return detection_to_bearing(bbox_center, bbox_size, object_size, K, D, K_inv)

    def point_to_bearing(self, bbox_center):
        K, D, K_inv = self.K, self.D, self.K_inv
        return point_to_bearing(bbox_center, K, D, K_inv)


def detection_to_bearing(bbox_center, bbox_size, object_size, K, D, K_inv):
    """Convert a 2D bounding box to a 3D unit-norm bearing vector based on object size"""
    x, y = bbox_center
    width_image, _ = bbox_size
    width_object, _, _ = object_size

    point_center = np.array([x, y])
    point_right = np.array([x + width_image / 2., y])

    bearing_center = point_to_bearing(point_center, K, D, K_inv)
    bearing_right = point_to_bearing(point_right, K, D, K_inv)

    angle = np.arccos(bearing_center.dot(bearing_right))
    distance = ((width_object / 2.) / np.tan(angle))

    bearing = bearing_center * distance
    return bearing


def point_to_bearing(bbox_center, K, D, K_inv):
    """Convert a 2D point in pixel coordinates to a unit-norm 3D bearing vector."""
    image_point_homogeneous = np.array([*bbox_center, 1.0])

    world_point = K_inv.dot(image_point_homogeneous)
    bearing_norm = world_point / np.linalg.norm(world_point)
    return bearing_norm


class RelativeLocalisationNode(Node):
    """ A simple subscriber node for YOLO detection and tracking """

    def __init__(self, namespace: str, camera_names: List[str], publish_markers: bool = False):
        super().__init__(f"{namespace}_yolo_subscriber")
        self.publish_markers = publish_markers
        self.namespace = namespace
        self.relative_localizers = {}
        self.camera_names = camera_names
        self.detections = {camera_name: None for camera_name in camera_names}
        self.all_detection = Detection3DArray()
        # self.detection_buffer = {camera_name: [] for camera_name in camera_names}
        # self.camera_timers = {}
        self.camera_indices = {camera_name: i for i, camera_name in enumerate(camera_names)}
        self.last_processed_times = {camera_name: 0 for camera_name in camera_names}
        self.detections_ready = {camera_name: False for camera_name in camera_names}

        for camera_name in camera_names:
            self.detection_subscription = self.create_subscription(
                DetectionArray,
                f"/{namespace}/{camera_name}/yolo/detections",
                lambda msg, camera_name=camera_name: self.yolo_detection_callback(msg, camera_name),
                10)
            print(f"YOLO subscriber for {namespace}/{camera_name} created")

        if self.publish_markers:
            print(f"Creating marker publisher for {namespace}")
            self.marker_pub = self.create_publisher(MarkerArray, f"/{namespace}/object_markers", 10)
            self.prev_markers = {}
            self.timer = self.create_timer(1.0, self.marker_timer_callback)

        # create publishers for the relative localizations for each camera
        for camera_name in camera_names:
            fx = 369.5
            fy = 369.5
            cx = 640.5
            cy = 480.5

            K = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]])

            D = np.array([])

            # Initialize RelativeLocalizer with camera matrix and distortion coefficients
            self.relative_localizers[camera_name] = RelativeLocalizer(K, D)

        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)

        detections_pub_topic = f"/{namespace}/detections"
        self.detections_pub = self.create_publisher(Detection3DArray, detections_pub_topic, 10)

        
        

    def marker_timer_callback(self):
        marker_array = MarkerArray()
        current_marker_ids = set()

        for detection in self.all_detection.detections:
            marker = Marker()
            marker.header.frame_id = f"{self.namespace}/base_link"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "object_markers"
            marker.id = len(current_marker_ids)
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            # marker.lifetime = Duration(sec=10, nanosec=0)

            # Set the marker position to the estimated object position
            marker.pose.position.x = 0.0
            marker.pose.position.y = 0.0
            marker.pose.position.z = 0.0

            # Set the marker orientation to point towards the object
            direction = np.array([detection.bbox.center.position.x,
                                detection.bbox.center.position.y,
                                detection.bbox.center.position.z])
            
            # print(f"Direction: {direction}")
            direction_norm = direction / np.linalg.norm(direction)
            # print(f"Direction norm: {direction_norm}")

                # Convert the direction vector to quaternion
            rot_matrix = np.eye(4)
            rot_matrix[:3, 0] = direction_norm
            rot_matrix[:3, 1] = np.cross(np.array([0, 0, 1]), direction_norm)
            rot_matrix[:3, 2] = np.cross(direction_norm, rot_matrix[:3, 1])

            # use scipy to convert the rotation matrix to quaternion
            
            r = R.from_matrix(rot_matrix[:3, :3])
            q = r.as_quat()
        
            marker.pose.orientation.x = q[0]
            marker.pose.orientation.y = q[1]
            marker.pose.orientation.z = q[2]
            marker.pose.orientation.w = q[3]

            # Set the marker scale (adjust as needed)
            distance = np.linalg.norm(direction)
            marker.scale.x = distance
            marker.scale.y = 0.1
            marker.scale.z = 0.1

            # Set the marker color (adjust as needed)
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0

            current_marker_ids.add(marker.id)
            marker_array.markers.append(marker)

        # Publish the marker array
        self.marker_pub.publish(marker_array)
        self.prev_markers = current_marker_ids

    def yolo_detection_callback(self, msg, camera_name):
        # print(f"YOLO detection callback for {camera_name}")
        current_time = self.get_clock().now().to_msg().sec
        last_processed_time = self.last_processed_times[camera_name]

        # if current_time - last_processed_time < 1:
        #     return

        out_detections_msg = Detection3DArray()
        out_detections_msg.header.stamp = self.get_clock().now().to_msg()

        # set frame id to the camera frame

        out_detections_msg.header.frame_id = f"{self.namespace}/{camera_name}"

        for i, detection in enumerate(msg.detections):

            if detection.class_name != 'airplane':
                continue

            localiser = self.relative_localizers[camera_name]

            # print(f"bbox center: {detection.bbox.center.position.x}, {detection.bbox.center.position.y}")

            bbox_center = np.array([detection.bbox.center.position.x, detection.bbox.center.position.y])
            bbox_size = (detection.bbox.size.x, detection.bbox.size.y)
            object_size = (0.67, 0.11, 0.67)  # quad base size with rotors (width, height, depth)

            bearing = localiser.detection_to_bearing(bbox_center, bbox_size, object_size)

            # print(f"Camera {camera_name} detected object at bearing {bearing}")

            source_frame = f"{self.namespace}/{camera_name}"  # drone_X/camera_X_optical
            target_frame = f"{self.namespace}/base_link"

        
            try:
                transform = self.buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time())
            except Exception as e:
                print(e)
                continue

            point = PointStamped()
            point.header.frame_id = source_frame
            point.header.stamp = self.get_clock().now().to_msg()
            point.point.x = bearing[2]
            point.point.y = -bearing[0]
            point.point.z = -bearing[1]

            # print(f"Original bearing: {point.point}, source frame: {source_frame}, target frame: {target_frame}")

            point_tf = tf2_geometry_msgs.do_transform_point(point, transform)

            # print(f"Transformed bearing: {point_tf.point}, source frame: {source_frame}, target frame: {target_frame}")


            out_detection = Detection3D()
            out_detection.header = out_detections_msg.header
            out_detection.bbox.center.position.x = point_tf.point.x
            out_detection.bbox.center.position.y = point_tf.point.y
            out_detection.bbox.center.position.z = point_tf.point.z

            out_detection.bbox.size.x = object_size[2]
            out_detection.bbox.size.y = object_size[0]
            out_detection.bbox.size.z = object_size[1]

            self.all_detection.detections.append(out_detection)

            # out_detections_msg.detections.append(out_detection)
            # self.detections[camera_name] = out_detections_msg.detections

        self.last_processed_times[camera_name] = current_time
        self.detections_ready[camera_name] = True

        if all(self.detections_ready.values()):
            self.all_detection.header.stamp = self.get_clock().now().to_msg()
            self.all_detection.header.frame_id = f"{self.namespace}/base_link"
            self.detections_pub.publish(self.all_detection)
            self.all_detection.detections.clear()
            self.detections_ready = {camera_name: False for camera_name in self.camera_names}

def main(args=None):
    rclpy.init(args=args)

    namespace_list = ["drone0", "drone1", "drone2"]  # Replace with your desired namespaces
    camera_names = ["left_camera", "front_camera", "right_camera", "rear_camera"]  # Replace with your desired camera names

    publish_markers = True

    if args is not None and len(args) > 1:
        if args[1] == "--markers":
            publish_markers = True

    yolo_subscribers = []
    for namespace in namespace_list:
        yolo_subscriber = RelativeLocalisationNode(namespace, camera_names, publish_markers=publish_markers)
        yolo_subscribers.append(yolo_subscriber)

    executor = rclpy.executors.MultiThreadedExecutor()
    for yolo_subscriber in yolo_subscribers:
        executor.add_node(yolo_subscriber)
    executor.spin()

    for yolo_subscriber in yolo_subscribers:
        yolo_subscriber.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()