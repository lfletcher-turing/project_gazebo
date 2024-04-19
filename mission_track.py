
import sys
import threading
from typing import List, Optional
from math import radians, cos, sin
from itertools import cycle, islice

import rclpy
from rclpy.node import Node
from rclpy.timer import Timer
from as2_msgs.msg import YawMode
from as2_msgs.msg import BehaviorStatus
from as2_python_api.drone_interface import DroneInterface
from as2_python_api.behavior_actions.behavior_handler import BehaviorHandler
import numpy as np
import cv2 as cv
from yolov8_msgs.msg import DetectionArray

class YOLOSubscriber(Node):
    """ A simple subscriber node for YOLO detection and tracking """
    
    def __init__(self, namespace: str):
        super().__init__(f"{namespace}_yolo_subscriber")
        
        self.tracking_subscription = self.create_subscription(
            DetectionArray,
            f"/{namespace}/yolo/tracking",
            self.yolo_tracking_callback,
            10)

        print(f"YOLO subscriber for {namespace} created")

        self.log_timer = None
        self.last_detection_msg = None
        self.last_tracking_msg = None

    def yolo_tracking_callback(self, msg):
        if self.is_dancing:
            self.last_tracking_msg = msg

    def log_callback(self):

        if self.last_tracking_msg is not None:
            """YOLO tracking callback"""
            if len(self.last_tracking_msg.detections) > 0:
                detection = self.last_tracking_msg.detections[0]  # Assuming you want to use the first detection
                bbox_center = detection.bbox.center.position
                bbox_size = detection.bbox.size
                
                self.get_logger().info(f"YOLO tracking triggered")
                self.get_logger().info(f"Bounding box center: x={bbox_center.x}, y={bbox_center.y}")
                self.get_logger().info(f"Bounding box size: x={bbox_size.x}, y={bbox_size.y}")
                self.get_logger().info(f"Bounding box class: {detection.class_name}")
                self.get_logger().info(f"Bounding box confidence: {detection.id}")
            else:
                self.get_logger().info(f"No tracking")

        self.last_detection_msg = None
        self.last_tracking_msg = None