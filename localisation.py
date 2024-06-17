#!/bin/python3

"""
localisation.py
"""

from typing import List
import rclpy
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from vision_msgs.msg import Detection3DArray, Detection3D, Detection2DArray, Detection2D, ObjectHypothesisWithPose
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
import numpy as np
from yolov8_msgs.msg import DetectionArray
import tf2_geometry_msgs


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
        if self.has_parameter('use_sim_time'):
            self.set_parameters([rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])
        else:
            self.declare_parameter('use_sim_time', True)
        use_sim_time = self.get_parameter('use_sim_time').value
        self.publish_markers = publish_markers
        self.namespace = namespace
        self.relative_localizers = {}
        self.camera_names = camera_names
        self.detections = {camera_name: None for camera_name in camera_names}
        self.all_detections_camera = Detection3DArray()
        self.all_detections_transformed = Detection3DArray()
        self.all_detections_raw = Detection2DArray()
        self.camera_indices = {camera_name: i for i, camera_name in enumerate(camera_names)}
        self.detections_ready = {camera_name: False for camera_name in camera_names}
        self.latest_detections_transformed = {camera_name: None for camera_name in camera_names}
        self.latest_detections_camera = {camera_name: None for camera_name in camera_names}
        self.latest_detections_raw = {camera_name: None for camera_name in camera_names}

        for camera_name in camera_names:
            self.detection_subscription = self.create_subscription(
                DetectionArray,
                f"/{namespace}/{camera_name}/yolo/detections",
                lambda msg, camera_name=camera_name: self.yolo_detection_callback(msg, camera_name),
                1)
            print(f"YOLO subscriber for {namespace}/{camera_name} created")

        if self.publish_markers:
            print(f"Creating marker publisher for {namespace}")
            self.marker_pub = self.create_publisher(MarkerArray, f"/{namespace}/object_markers", 10)
            self.prev_markers = {}
            # self.timer = self.create_timer(1.0, self.marker_timer_callback)

        # create publishers for the relative localizations for each camera
        for camera_name in camera_names:
            fx = 184.75
            fy = 184.75
            cx = 320.5
            cy = 240.5

            K = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]])

            D = np.array([])

            # Initialize RelativeLocalizer with camera matrix and distortion coefficients
            self.relative_localizers[camera_name] = RelativeLocalizer(K, D)

        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)

        raw_detections_pub_topic = f"/{namespace}/raw_detections"
        self.raw_detections_pub = self.create_publisher(Detection2DArray, raw_detections_pub_topic, 1)

        camera_detections_pub_topic = f"/{namespace}/camera_detections"
        self.camera_detections_pub = self.create_publisher(Detection3DArray, camera_detections_pub_topic, 1)

        transformed_detections_pub_topic = f"/{namespace}/transformed_detections"
        self.transformed_detections_pub = self.create_publisher(Detection3DArray, transformed_detections_pub_topic, 1)

    def do_publish_markers(self):
        marker_array = MarkerArray()
        current_marker_ids = set()

        # check and delete old markers
        for marker_id in self.prev_markers:
            marker = Marker()
            marker.header.frame_id = f"{self.namespace}/base_link"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "object_markers"
            marker.id = marker_id
            marker.action = Marker.DELETE
            marker_array.markers.append(marker)

        for detection in self.all_detections_transformed.detections:
            # print(f"Publishing marker for object at {detection.bbox.center.position.x}, {detection.bbox.center.position.y}, {detection.bbox.center.position.z}")
            marker = Marker()
            marker.header.frame_id = f"{self.namespace}/base_link"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "object_markers"
            marker.id = len(current_marker_ids)
            # print(f"Marker ID: {marker.id}")
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

        # if current_time - last_processed_time < 1:
        #     return


        # set frame id to the camera frame

        raw_detection_list = []
        camera_detections_list = []
        transformed_detections_list = []

        for detection in (msg.detections):

        
            if detection.class_name != 'airplane':
                continue
            # consolidate raw detections into a single message and publish

            raw_detection = Detection2D()
            raw_detection.header.frame_id = f"{self.namespace}/{camera_name}"
            raw_detection.header.stamp = self.get_clock().now().to_msg()
            raw_detection.bbox.center.position.x = detection.bbox.center.position.x
            raw_detection.bbox.center.position.y = detection.bbox.center.position.y
            raw_detection.bbox.size_x = detection.bbox.size.x
            raw_detection.bbox.size_y = detection.bbox.size.y
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = detection.class_name
            hypothesis.hypothesis.score = detection.score
            raw_detection.results.append(hypothesis)

            raw_detection_list.append(raw_detection)

            

            localiser = self.relative_localizers[camera_name]

            # print(f"bbox center: {detection.bbox.center.position.x}, {detection.bbox.center.position.y}")

            bbox_center = np.array([detection.bbox.center.position.x, detection.bbox.center.position.y])
            bbox_size = (detection.bbox.size.x, detection.bbox.size.y)
            object_size = (0.67, 0.11, 0.67)  # quad base size with rotors (width, height, depth) # TODO put in config file 

            bearing = localiser.detection_to_bearing(bbox_center, bbox_size, object_size)

            # print(f"Camera {camera_name} detected object at bearing {bearing}")

            source_frame = f"{self.namespace}/{camera_name}"  # drone_X/camera_X_optical
            target_frame = f"{self.namespace}/base_link"
        
            try:
                transform = self.buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time())
            except Exception as e:
                print(e)
                continue
            
            time_stamp = self.get_clock().now().to_msg()

            # print(time_stamp)

            point = PointStamped()
            point.header.frame_id = source_frame
            point.header.stamp = time_stamp
            point.point.x = bearing[2]
            point.point.y = -bearing[0]
            point.point.z = -bearing[1]

            camera_detection = Detection3D()
            camera_detection.header.frame_id = source_frame
            camera_detection.header.stamp = time_stamp
            camera_detection.bbox.center.position.x = point.point.x
            camera_detection.bbox.center.position.y = point.point.y
            camera_detection.bbox.center.position.z = point.point.z

            camera_detection.bbox.size.x = object_size[2]
            camera_detection.bbox.size.y = object_size[0]
            camera_detection.bbox.size.z = object_size[1]

            camera_detections_list.append(camera_detection)
    
            # print(f"Original bearing: {point.point}, source frame: {source_frame}, target frame: {target_frame}")

            point_tf = tf2_geometry_msgs.do_transform_point(point, transform)

            # print(f"Transformed bearing: {point_tf.point}, source frame: {source_frame}, target frame: {target_frame}")

            transformed_detection = Detection3D()

            transformed_detection.bbox.center.position.x = point_tf.point.x
            transformed_detection.bbox.center.position.y = point_tf.point.y
            transformed_detection.bbox.center.position.z = point_tf.point.z
            transformed_detection.bbox.size.x = object_size[2]
            transformed_detection.bbox.size.y = object_size[0]
            transformed_detection.bbox.size.z = object_size[1]
            transformed_detection.header.frame_id = source_frame
            transformed_detection.header.stamp = time_stamp

            transformed_detections_list.append(transformed_detection)

        self.latest_detections_transformed[camera_name] = transformed_detections_list
        self.latest_detections_camera[camera_name] = camera_detections_list
        self.latest_detections_raw[camera_name] = raw_detection_list
        self.detections_ready[camera_name] = True

        if all(self.detections_ready.values()):
            self.publish_detections_callback()


    def publish_detections_callback(self):
        self.all_detections_camera.detections.clear()
        self.all_detections_transformed.detections.clear()
        self.all_detections_raw.detections.clear()
        # if there are new detections, add them to the all_detections list, otherwise, publish the existing list
        for camera_name in self.camera_names:
            self.detections_ready[camera_name] = False
            try:
                self.all_detections_camera.detections.extend(self.latest_detections_camera[camera_name])
                self.all_detections_transformed.detections.extend(self.latest_detections_transformed[camera_name])
                self.all_detections_raw.detections.extend(self.latest_detections_raw[camera_name])
            except Exception as e:
                continue

        time_now = self.get_clock().now().to_msg()
        self.all_detections_transformed.header.stamp = time_now
        self.all_detections_transformed.header.frame_id = f"{self.namespace}/base_link"
        self.transformed_detections_pub.publish(self.all_detections_transformed)

        if self.publish_markers:
            self.do_publish_markers()
        # 
    
        # self.all_detections_transformed.detections.clear()

        self.all_detections_camera.header.stamp = time_now
        self.all_detections_camera.header.frame_id = f"{self.namespace}/base_link"
        self.camera_detections_pub.publish(self.all_detections_camera)

        self.all_detections_raw.header.stamp = time_now
        self.all_detections_raw.header.frame_id = f"{self.namespace}/base_link"
        self.raw_detections_pub.publish(self.all_detections_raw)
        # self.all_detections_camera.detections.clear()





def main(args=None):
    rclpy.init(args=args)

    namespace_list = ["drone0", "drone1", "drone2"]  # Replace with your desired namespaces
    camera_names = ["left_camera", "front_camera", "right_camera", "rear_camera"]  # Replace with your desired camera names


    if args is not None and len(args) > 1:
        if args[1] == "--markers":
            publish_markers = True
    else:
        publish_markers = False

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