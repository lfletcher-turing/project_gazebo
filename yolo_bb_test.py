import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class YoloSubscriber(Node):
    def __init__(self):
        super().__init__('yolo_subscriber')
        self.subscription = self.create_subscription(
            BoundingBoxes,
            '/yolov8/bounding_boxes',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()
        self.timer = self.create_timer(1.0, self.timer_callback)

    def listener_callback(self, msg):
        self.bounding_boxes = msg.bounding_boxes

    def timer_callback(self):
        if hasattr(self, 'bounding_boxes'):
            for box in self.bounding_boxes:
                print(f"Class: {box.class_id}, Confidence: {box.probability}")
                print(f"Bounding Box: (x: {box.xmin}, y: {box.ymin}), (x: {box.xmax}, y: {box.ymax})")
                print("---")
        else:
            print("No bounding boxes received yet.")

def main(args=None):
    rclpy.init(args=args)
    yolo_subscriber = YoloSubscriber()
    rclpy.spin(yolo_subscriber)
    yolo_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()