#!/bin/bash

source utils/tools.bash

# Get drone namespaces from command-line argument
drone_namespaces_string=$1
drone_namespaces=($(string_to_list "$drone_namespaces_string"))

echo "Recording rosbag for drones: ${drone_namespaces[@]}"

# Create directory for rosbags
mkdir rosbag/rosbags 2>/dev/null
cd rosbag/rosbags

# Construct the rosbag record command
rosbag_cmd="ros2 bag record"

# Add topics and drone namespaces to the rosbag record command
# for drone_namespace in "${drone_namespaces[@]}"; do
#   rosbag_cmd+=" /${drone_namespace}/platform/info \
#                 /${drone_namespace}/self_localization/pose \
#                 /${drone_namespace}/self_localization/twist \
#                 /${drone_namespace}/actuator_command/twist \
#                 /${drone_namespace}/filtered_detections \
#                 /${drone_namespace}/migration_goal"
# done

for drone_namespace in "${drone_namespaces[@]}"; do
  rosbag_cmd+=" /${drone_namespace}/platform/info \
                /metrics/${drone_namespace}/pose \
                /metrics/${drone_namespace}/twist \
                /metrics/${drone_namespace}/actuator_command \
                /metrics/${drone_namespace}/filtered_detections \
                /metrics/${drone_namespace}/transformed_detections \
                /metrics/${drone_namespace}/camera_detections \
                /metrics/${drone_namespace}/object_markers \
                /metrics/${drone_namespace}/estimated_poses"
done

# Add remaining topics
rosbag_cmd+=" /tf /tf_static /metrics/swarm/migration_waypoint /metrics/swarm/agent_distances_mean /metrics/swarm/agent_distances_std /metrics/swarm/agent_distances_min /metrics/swarm/agent_distances_max --include-hidden-topics"

# Execute the rosbag record command
eval "$rosbag_cmd"