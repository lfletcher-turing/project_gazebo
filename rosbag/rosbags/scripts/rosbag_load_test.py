import pandas as pd
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
from rosbags.typesys.types import std_msgs__msg__Float64, geometry_msgs__msg__PoseStamped, geometry_msgs__msg__Twist
import re

# Initialize type store
typestore = get_typestore(Stores.LATEST)

# Define a function to process each message type
def process_message(msg, msgtype):
    if msgtype == 'std_msgs/msg/Float64':
        return {'value': msg.data}
    elif msgtype == 'geometry_msgs/msg/PoseStamped':
        return {
            'position_x': msg.pose.position.x,
            'position_y': msg.pose.position.y,
            'position_z': msg.pose.position.z,
            'orientation_x': msg.pose.orientation.x,
            'orientation_y': msg.pose.orientation.y,
            'orientation_z': msg.pose.orientation.z,
            'orientation_w': msg.pose.orientation.w
        }
    elif msgtype == 'geometry_msgs/msg/Twist':
        return {
            'linear_x': msg.twist.linear.x,
            'linear_y': msg.twist.linear.y,
            'linear_z': msg.twist.linear.z,
            'angular_x': msg.twist.angular.x,
            'angular_y': msg.twist.angular.y,
            'angular_z': msg.twist.angular.z
        }
    # Add more processing logic for other message types if needed
    else:
        return None

# Function to extract drone number from topic or return 'swarm' if found
def get_drone_number(topic):
    if 'swarm' in topic:
        return 'swarm'
    match = re.search(r'drone(\d+)', topic)
    if match:
        return int(match.group(1))
    return None

# Initialize a dictionary to store data
data_dict = {
    'timestamp': [],
    'topic': [],
    'drone_number': [],
    'data': []
}

# Open the ROS bag for reading
with Reader('/home/azureuser/project_gazebo/rosbag/rosbags/rosbag2_2024_05_23-14_14_47') as reader:
    # Iterate over messages
    for connection, timestamp, rawdata in reader.messages():
        try:
            msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
            processed_data = process_message(msg, connection.msgtype)
            
            if processed_data is not None:
                data_dict['timestamp'].append(timestamp)
                data_dict['topic'].append(connection.topic)
                data_dict['drone_number'].append(get_drone_number(connection.topic))
                data_dict['data'].append(processed_data)
        except Exception as e:
            continue

# Convert the dictionary to a pandas DataFrame
df = pd.DataFrame(data_dict)

# Normalize the data column which contains dictionaries
df = df.join(pd.json_normalize(df.pop('data')))

# Display the DataFrame
print(df)