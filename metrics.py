import argparse
import csv
from typing import List
import rclpy
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import PointStamped, Point, Pose, Twist, PoseStamped, PoseArray
from visualization_msgs.msg import Marker, MarkerArray
from vision_msgs.msg import Detection3DArray, Detection3D, Detection2DArray, Detection2D
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
import numpy as np
from yolov8_msgs.msg import DetectionArray
import tf2_geometry_msgs
from std_msgs.msg import Float64
from scipy.spatial.distance import pdist
import time

class Metrics(Node):

    def __init__(self, namespaces: List[str], publish_rate: float = 1, publish_namespace: str = 'metrics', save_csv: bool = False, vision_msgs: bool = True, publish = True):
        super().__init__('metrics_node')
        if self.has_parameter('use_sim_time'):
            self.set_parameters([rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])
        else:
            self.declare_parameter('use_sim_time', True)
        self.get_logger().info('Initializing Metrics Node')

        self.namespaces = namespaces
        self.publish_rate = publish_rate

        self.latest_messages = {}
        self.messages_updated ={}


        self.save_csv = save_csv
        self.vision_msgs = vision_msgs
        self.publish = publish

        raw_detection_headers = []


        # camera, bboxx, bboxy, sizex, sizey, score, aspect_ratio

        for i in range(5):
            raw_detection_headers.append(f'raw_detection_{i+1}_camera')
            raw_detection_headers.append(f'raw_detection_{i+1}_bbox_x')
            raw_detection_headers.append(f'raw_detection_{i+1}_bbox_y')
            raw_detection_headers.append(f'raw_detection_{i+1}_size_x')
            raw_detection_headers.append(f'raw_detection_{i+1}_size_y')
            raw_detection_headers.append(f'raw_detection_{i+1}_score')
            raw_detection_headers.append(f'raw_detection_{i+1}_aspect_ratio')

        if self.save_csv:
            if self.vision_msgs:
                self.csv_header = [
                    'timestamp', 'namespace', 'position_x', 'position_y', 'position_z',
                    'orientation_x', 'orientation_y', 'orientation_z', 'orientation_w',
                    'linear_x', 'linear_y', 'linear_z', 'angular_x', 'angular_y',
                    'angular_z', 'average_distance', 'std_distance', 'min_distance',
                    'max_distance', 'filtered_position_1_x', 'filtered_position_1_y',
                    'filtered_position_1_z', 'filtered_position_2_x', 'filtered_position_2_y',
                    'filtered_position_2_z', 'filtered_position_3_x', 'filtered_position_3_y',
                    'filtered_position_3_z', 'estimated_position_1_x', 'estimated_position_1_y',
                    'estimated_position_1_z', 'estimated_position_2_x', 'estimated_position_2_y',
                    'estimated_position_2_z', 'estimated_position_3_x', 'estimated_position_3_y',
                    'estimated_position_3_z', 'estimated_position_4_x', 'estimated_position_4_y',
                    'estimated_position_4_z', 'estimated_position_5_x', 'estimated_position_5_y',
                    'estimated_position_5_z', 'migration_waypoint_x', 'migration_waypoint_y',
                    'migration_waypoint_z']
                self.csv_header.extend(raw_detection_headers)
                
            else:
                self.csv_header = [
                    'timestamp', 'namespace', 'position_x', 'position_y', 'position_z',
                    'orientation_x', 'orientation_y', 'orientation_z', 'orientation_w',
                    'linear_x', 'linear_y', 'linear_z', 'angular_x', 'angular_y',
                    'angular_z', 'average_distance', 'std_distance', 'min_distance',
                    'max_distance', 'migration_waypoint_x', 'migration_waypoint_y',
                    'migration_waypoint_z']

            if self.vision_msgs:
                
                self.csv_writer_funcs = {
                    'pose': self.pose_csv_row, 'twist': self.twist_csv_row,
                    'detection': self.detection_csv_row,
                    'estimated_poses': self.estimated_pose_csv_row,
                    'migration_waypoint': self.migration_waypoint_csv_row,
                    'raw_detections': self.raw_detection_csv_row}
            else:
                self.csv_writer_funcs = {'pose': self.pose_csv_row, 'twist': self.twist_csv_row, 'migration_waypoint': self.migration_waypoint_csv_row}

            # create filename with todays data and time
            file_stamp = time.strftime('%Y-%m-%d-%H-%M-%S')
            self.csv_filename = f'./metric_logs/metrics_{file_stamp}.csv' 
            self.csv_file = open(self.csv_filename, 'w', newline='')
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.csv_header)
            self.csv_writer.writeheader()

        self.camera_names = ['front_camera', 'left_camera', 'right_camera', 'rear_camera']

        for ns in namespaces:
            self.latest_messages[ns] = {
                'camera_detections': None,
                'transformed_detections': None,
                'filtered_detections': None,
                'pose': None,
                'twist': None,
                'actuator_command': None,
                'object_markers': None,
                'estimated_poses_from_detections': None,
                'raw_detections': None
            }

        self.latest_messages['swarm'] = {
            'migration_waypoint': None,
            'agent_distances_mean': None,
            'agent_distances_std': None,
            'agent_distances_min': None,
            'agent_distances_max': None
        }
        self.messages_updated = {
            ns: {'camera_detections': False, 'transformed_detections': False,
                  'filtered_detections': False, 'pose': False, 'twist': False, 'actuator_command': False, 'object_markers': False, 'estimated_poses_from_detections': False, 'raw_detections': False}
                    for ns in namespaces}
        # print(f"namespaces: {self.namespaces}")
        # print(f"messages updated: {self.messages_updated}")

        self.messages_updated['swarm'] = {key: False for key in self.latest_messages['swarm'].keys()}

        # print(f"messages updated: {self.messages_updated}")
        if self.vision_msgs:
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscribers(namespaces)
        if self.publish:
            self.create_publishers()
        self.create_timer(1.0 / self.publish_rate, self.publish_metrics)


    def raw_detection_csv_row(self, detections):
        if detections:
            raw_detection_dict = {}
            for i, detection in enumerate(detections.detections):
                raw_detection_dict[f'raw_detection_{i+1}_camera'] = detection.header.frame_id.split('/')[1]
                raw_detection_dict[f'raw_detection_{i+1}_bbox_x'] = detection.bbox.center.position.x
                raw_detection_dict[f'raw_detection_{i+1}_bbox_y'] = detection.bbox.center.position.y
                raw_detection_dict[f'raw_detection_{i+1}_size_x'] = detection.bbox.size_x
                raw_detection_dict[f'raw_detection_{i+1}_size_y'] = detection.bbox.size_y
                raw_detection_dict[f'raw_detection_{i+1}_score'] = detection.results[0].hypothesis.score
                raw_detection_dict[f'raw_detection_{i+1}_aspect_ratio'] = detection.bbox.size_x / detection.bbox.size_y
            if len(detections.detections) < 5:
                for i in range(len(detections.detections), 5):
                    raw_detection_dict[f'raw_detection_{i+1}_camera'] = np.nan
                    raw_detection_dict[f'raw_detection_{i+1}_bbox_x'] = np.nan
                    raw_detection_dict[f'raw_detection_{i+1}_bbox_y'] = np.nan
                    raw_detection_dict[f'raw_detection_{i+1}_size_x'] = np.nan
                    raw_detection_dict[f'raw_detection_{i+1}_size_y'] = np.nan
                    raw_detection_dict[f'raw_detection_{i+1}_score'] = np.nan
                    raw_detection_dict[f'raw_detection_{i+1}_aspect_ratio'] = np.nan
            return raw_detection_dict
        else:
            raw_detection_dict = {}
            for i in range(5):
                raw_detection_dict[f'raw_detection_{i+1}_camera'] = np.nan
                raw_detection_dict[f'raw_detection_{i+1}_bbox_x'] = np.nan
                raw_detection_dict[f'raw_detection_{i+1}_bbox_y'] = np.nan
                raw_detection_dict[f'raw_detection_{i+1}_size_x'] = np.nan
                raw_detection_dict[f'raw_detection_{i+1}_size_y'] = np.nan
                raw_detection_dict[f'raw_detection_{i+1}_score'] = np.nan
                raw_detection_dict[f'raw_detection_{i+1}_aspect_ratio'] = np.nan
            return raw_detection_dict

    def pose_csv_row(self, pose):
        if pose:
            return {
                'position_x': pose.pose.position.x,
                'position_y': pose.pose.position.y,
                'position_z': pose.pose.position.z,
                'orientation_x': pose.pose.orientation.x,
                'orientation_y': pose.pose.orientation.y,
                'orientation_z': pose.pose.orientation.z,
                'orientation_w': pose.pose.orientation.w
            }
        else:
            return {
                'position_x': np.nan,
                'position_y': np.nan,
                'position_z': np.nan,
                'orientation_x': np.nan,
                'orientation_y': np.nan,
                'orientation_z': np.nan,
                'orientation_w': np.nan
            }
    
    def twist_csv_row(self, twist):
        if twist:
            return {
                'linear_x': twist.linear.x,
                'linear_y': twist.linear.y,
                'linear_z': twist.linear.z,
                'angular_x': twist.angular.x,
                'angular_y': twist.angular.y,
                'angular_z': twist.angular.z
            }
        else:
            return {
                'linear_x': np.nan,
                'linear_y': np.nan,
                'linear_z': np.nan,
                'angular_x': np.nan,
                'angular_y': np.nan,
                'angular_z': np.nan
            }
    
    def detection_csv_row(self, detections):
        if detections:
            if len(detections.detections) > 0:
                for i, detection in enumerate(detections.detections):
                    return {
                        f'filtered_position_{i+1}_x': detection.bbox.center.position.x,
                        f'filtered_position_{i+1}_y': detection.bbox.center.position.y,
                        f'filtered_position_{i+1}_z': detection.bbox.center.position.z
                    }
                if len(detections.detections) < 3:
                    for i in range(len(detections.detections), 3):
                        return {
                            f'filtered_position_{i+1}_x': np.nan,
                            f'filtered_position_{i+1}_y': np.nan,
                            f'filtered_position_{i+1}_z': np.nan
                        }
      
        
    def estimated_pose_csv_row(self, poses):
        # print(f"Poses 1: {poses}")
        if poses:
            esimated_poses_dict = {}
            # print(f"Poses 2: {poses.poses}")
            for i, pose in enumerate(poses.poses):
                esimated_poses_dict[f'estimated_position_{i+1}_x'] = pose.position.x
                esimated_poses_dict[f'estimated_position_{i+1}_y'] = pose.position.y
                esimated_poses_dict[f'estimated_position_{i+1}_z'] = pose.position.z
            if len(poses.poses) < 5:
                for i in range(len(poses.poses), 5):
                    esimated_poses_dict[f'estimated_position_{i+1}_x'] = np.nan
                    esimated_poses_dict[f'estimated_position_{i+1}_y'] = np.nan
                    esimated_poses_dict[f'estimated_position_{i+1}_z'] = np.nan
        else:
            esimated_poses_dict = {}
            for i in range(5):
                esimated_poses_dict[f'estimated_position_{i+1}_x'] = np.nan
                esimated_poses_dict[f'estimated_position_{i+1}_y'] = np.nan
                esimated_poses_dict[f'estimated_position_{i+1}_z'] = np.nan

        # print(f"Estimated Poses Dict: {esimated_poses_dict}")
        return esimated_poses_dict
        
    def migration_waypoint_csv_row(self, waypoint):
        # print(f"Migration Waypoint: {waypoint}")
        if waypoint:
            return {
                'migration_waypoint_x': waypoint.x,
                'migration_waypoint_y': waypoint.y,
                'migration_waypoint_z': waypoint.z
            }
        else:
            return {
                'migration_waypoint_x': np.nan,
                'migration_waypoint_y': np.nan,
                'migration_waypoint_z': np.nan
            }


    def create_subscribers(self, namespaces: List[str]):
        # subscribe to namespace topics
        # detection topics
        if self.vision_msgs:
            for ns in namespaces:
                self.create_subscription(Detection3DArray, f'/{ns}/camera_detections', self.create_callback(ns, 'camera_detections'), 1)
                self.create_subscription(Detection3DArray, f'{ns}/transformed_detections', self.create_callback(ns, 'transformed_detections'), 1)
                self.create_subscription(Detection3DArray, f'/{ns}/filtered_detections', self.create_callback(ns, 'filtered_detections' ), 1)
                self.create_subscription(MarkerArray, f'/{ns}/object_markers', self.create_callback(ns, 'object_markers'), 1)
                self.create_subscription(Detection2DArray, f'{ns}/raw_detections', self.create_callback(ns, 'raw_detections'), 1)

        # pose and twist topics
        for ns in namespaces:
            qos_profile = rclpy.qos.QoSProfile(reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT, depth=1, durability=rclpy.qos.DurabilityPolicy.VOLATILE)
            self.create_subscription(PoseStamped, f'/{ns}/self_localization/pose', self.create_callback(ns, 'pose'), qos_profile)
            self.create_subscription(Twist, f'/{ns}/self_localization/twist', self.create_callback(ns, 'twist'), qos_profile)

        for ns in namespaces:
            self.create_subscription(Twist, f'/{ns}/actuator_command/twist', self.create_callback(ns, 'actuator_command'), 1)
        
        # migration waypoint topic
        self.create_subscription(Point, '/drone0/migration_waypoint', self.create_callback('swarm', 'migration_waypoint'), 1)

    def create_publishers(self):

        # create publishers for metrics
        self.metric_publishers = {}
        for ns in self.namespaces:
            self.metric_publishers[ns] = {
                'camera_detections': self.create_publisher(Detection3DArray, f'/metrics/{ns}/camera_detections', 10),
                'transformed_detections': self.create_publisher(Detection3DArray, f'/metrics/{ns}/transformed_detections', 10),
                'filtered_detections': self.create_publisher(Detection3DArray, f'/metrics/{ns}/filtered_detections', 10),
                'pose': self.create_publisher(PoseStamped, f'/metrics/{ns}/pose', 10),
                'twist': self.create_publisher(Twist, f'/metrics/{ns}/twist', 10),
                'actuator_command': self.create_publisher(Twist, f'/metrics/{ns}/actuator_command', 10),
                'object_markers': self.create_publisher(MarkerArray, f'/metrics/{ns}/object_markers', 10),
                'estimated_poses_from_detections': self.create_publisher(PoseArray, f'/metrics/{ns}/estimated_poses', 10),
                'raw_detections': self.create_publisher(Detection2DArray, f'/metrics/{ns}/raw_detections', 10)
            }

        self.metric_publishers['swarm'] = {
            'migration_waypoint': self.create_publisher(Point, '/metrics/swarm/migration_waypoint', 10),
            'agent_distances_mean': self.create_publisher(Float64, '/metrics/swarm/agent_distances_mean', 10),
            'agent_distances_std': self.create_publisher(Float64, '/metrics/swarm/agent_distances_std', 10),
            'agent_distances_min': self.create_publisher(Float64, '/metrics/swarm/agent_distances_min', 10),
            'agent_distances_max': self.create_publisher(Float64, '/metrics/swarm/agent_distances_max', 10)
        }


    def create_callback(self, namespace: str, topic: str):
        def callback(msg):
            self.latest_messages[namespace][topic] = msg
            self.messages_updated[namespace][topic] = True
        return callback
    
    def create_swarm_callback(self, topic: str):
        def callback(msg):
            self.latest_messages['swarm'][topic] = msg
            self.messages_updated['swarm'][topic] = True
        return callback

        
    def calculate_distance_metrics(self):

        # get positions of all agents
        positions = []
        for ns in self.namespaces:
            if self.latest_messages[ns]['pose']:
                pose = self.latest_messages[ns]['pose']
                positions.append(self.pose_to_array(pose.pose))

        # print(f'Positions: {positions}')

        if len(positions) < 2:
            return 0.0, 0.0, 0.0, 0.0

        distances = pdist(positions)

        return np.mean(distances), np.std(distances), np.min(distances), np.max(distances)
    
    def calculated_detected_agent_poses(self, ns: str):
        # print(f"Calculating Detected Agent Poses for {ns}")
        if self.latest_messages[ns]['filtered_detections']:
            detections = self.latest_messages[ns]['filtered_detections']
            # print(f" Number of detections: {len(detections.detections)}")
        
            poses = PoseArray()
            for detection in detections.detections:
                # print(f"Detection: {detection.bbox.center.position.x}")
                drone_frame = f'{ns}/base_link'
                world_frame = f'{ns}/odom'
                try:
                    # print(f"Time: {rclpy.time.Time().to_msg().sec}")
                    # print(f"Sim Time: {self.get_clock().now().to_msg().sec}")
                    # print(f"Drone Frame time: {self.tf_buffer.get_latest_common_time(world_frame, drone_frame)}")
                    transform = self.tf_buffer.lookup_transform(world_frame, drone_frame, rclpy.time.Time())
                    pose = PoseStamped()
                    pose.header.frame_id = drone_frame
                    pose.pose.position= Point(x=detection.bbox.center.position.x, y=detection.bbox.center.position.y, z=detection.bbox.center.position.z)
                    # print(f"Pose: {pose}")
                    transformed_pose = tf2_geometry_msgs.do_transform_pose(pose.pose, transform)
                    poses.poses.append(transformed_pose)
                    # print(f"Transformed Pose: {transformed_pose}")
                except Exception as e:
                    self.get_logger().error(f'Error transforming pose: {e}')
            return poses
        else:
            return None
                        

    def pose_to_array(self, pose: Pose):
        return [pose.position.x, pose.position.y]
    
    def publish_metrics(self):

        timestamp = self.get_clock().now().to_msg().sec * 1e9 + self.get_clock().now().to_msg().nanosec # convert to nanoseconds


        metrics_dicts = {}
        for ns in self.namespaces:
            if self.save_csv:
                metrics_dicts[ns] = {'timestamp': timestamp, 'namespace': ns}
            for key, publisher in self.metric_publishers[ns].items():
                # print(f'Publishing {ns}/{key}') 
                if self.messages_updated[ns][key]:
                    publisher.publish(self.latest_messages[ns][key])
                    self.messages_updated[ns][key] = False
                if self.save_csv:
                            # if key is in csv_writer_funcs, call the function to get the row data, then update the metrics_dicts, else ignore
                    if key in self.csv_writer_funcs:
                        metrics_dicts[ns].update(self.csv_writer_funcs[key](self.latest_messages[ns][key]))
                
                            

        mean, std, min, max = self.calculate_distance_metrics()

        self.metric_publishers['swarm']['agent_distances_mean'].publish(Float64(data=mean))
        self.metric_publishers['swarm']['agent_distances_std'].publish(Float64(data=std))
        self.metric_publishers['swarm']['agent_distances_min'].publish(Float64(data=min))
        self.metric_publishers['swarm']['agent_distances_max'].publish(Float64(data=max))
        if self.messages_updated['swarm']['migration_waypoint']:
            self.metric_publishers['swarm']['migration_waypoint'].publish(self.latest_messages['swarm']['migration_waypoint'])
        if self.save_csv:
            metrics_dicts['swarm'] = {'timestamp': timestamp, 'namespace': 'swarm'}
            metrics_dicts['swarm'].update(self.csv_writer_funcs['migration_waypoint'](self.latest_messages['swarm']['migration_waypoint']))
            metrics_dicts['swarm']['average_distance'] = mean
            metrics_dicts['swarm']['std_distance'] = std
            metrics_dicts['swarm']['min_distance'] = min
            metrics_dicts['swarm']['max_distance'] = max

        if self.vision_msgs:
            for ns in self.namespaces:
                # print(f"Estimating Poses for {ns}")
                estimated_poses = self.calculated_detected_agent_poses(ns)

                # print(f"Estimated Poses: {estimated_poses}")
                if estimated_poses:
                    self.metric_publishers[ns]['estimated_poses_from_detections'].publish(estimated_poses)
                if self.save_csv:
                    if estimated_poses:
                        # print(estimated_poses)
                        metrics_dicts[ns].update(self.csv_writer_funcs['estimated_poses'](estimated_poses))
                        # print the estimated poses
                        # print(f"metric dict : {metrics_dicts[ns]}")
                        # metrics_dicts[ns].update(self.csv_writer_funcs['estimated_poses']([]))
        
        # print(f"Metrics: {metrics_dicts}")

        if self.save_csv:
            for ns, metrics_dict in metrics_dicts.items():
                self.csv_writer.writerow(metrics_dict)
            self.csv_file.flush()



            


def main(args=None):
    rclpy.init(args=args)
    namespaces = ['drone0', 'drone1', 'drone2']  # Replace with your actual namespaces
    node = Metrics(namespaces, publish_rate=5.0, save_csv=True, vision_msgs=True)  # Replace with your desired publish rate
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--vision_msgs', action='store_true', help='Use GPS for flocking')
    # parser.add_argument('--save_csv', action='store_true', help='Use bagging')

    # args = parser.parse_args()
    main()





