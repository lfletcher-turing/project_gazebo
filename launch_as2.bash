#!/bin/bash

usage() {
    echo "  options:"
    echo "      -b: launch behavior tree"
    echo "      -m: multi agent"
    echo "      -r: record rosbag"
    echo "      -t: launch keyboard teleoperation"
    echo "      -n: drone namespace, default is drone0"
    echo "      -y: launch yolo"
    echo "      -f: launch foxglove"
    echo "      -s: headless gazebo"
    echo 
}

# Arg parser
while getopts "bmrtnyfs" opt; do
  case ${opt} in
    b )
      behavior_tree="true"
      ;;
    m )
      swarm="true"
      ;;
    r )
      record_rosbag="true"
      ;;
    t )
      launch_keyboard_teleop="true"
      ;;
    n )
      drone_namespace="${OPTARG}"
      ;;
    y )
      yolo="true"
      ;;
    f )
      foxglove="true"
      ;;
    s )
      headless="true"
      ;;
    \? )
      echo "Invalid option: -$OPTARG" >&2
      usage
      exit 1
      ;;
    : )
      if [[ ! $OPTARG =~ ^[wrt]$ ]]; then
        echo "Option -$OPTARG requires an argument" >&2
        usage
        exit 1
      fi
      ;;
  esac
done

source utils/tools.bash
source ~/yolo_ws/install/setup.bash

# Shift optional args
shift $((OPTIND -1))

## DEFAULTS
behavior_tree=${behavior_tree:="false"}
swarm=${swarm:="false"}
yolo=${yolo:="false"}
record_rosbag=${record_rosbag:="false"}
launch_keyboard_teleop=${launch_keyboard_teleop:="false"}
drone_namespace=${drone_namespace:="drone"}
foxglove=${foxglove:="false"}
headless=${headless:="false"}

if [[ ${swarm} == "true" ]]; then
  simulation_config="sim_config/world_swarm_line.json"
  num_drones=3
else
  simulation_config="sim_config/world.json" 
  num_drones=1
fi


# Generate the list of drone namespaces
drone_ns=()
for ((i=0; i<${num_drones}; i++)); do
  drone_ns+=("$drone_namespace$i")
done

camera_ns=("front_camera" "rear_camera" "left_camera" "right_camera")

for ns in "${drone_ns[@]}"
do
  tmuxinator start -n ${ns} -p tmuxinator/session.yml drone_namespace=${ns} simulation_config=${simulation_config} behavior_tree=${behavior_tree} &
  wait
done

if [[ ${record_rosbag} == "true" ]]; then
  tmuxinator start -n rosbag -p tmuxinator/rosbag.yml drone_namespace=$(list_to_string "${drone_ns[@]}") &
  wait
fi

if [[ ${launch_keyboard_teleop} == "true" ]]; then
  tmuxinator start -n keyboard_teleop -p tmuxinator/keyboard_teleop.yml simulation=true drone_namespace=$(list_to_string "${drone_ns[@]}") &
  wait
fi

if [[ ${yolo} == "true" ]]; then
for ns in "${drone_ns[@]}"
do
  for camera in "${camera_ns[@]}"
  do
    tmuxinator start -n ${ns}_${camera}_yolo -p tmuxinator/yolo.yml drone_namespace=${ns} camera_namespace=${camera} &
    wait
  done
done
fi

tmuxinator start -n gazebo -p tmuxinator/gazebo.yml simulation_config=${simulation_config} headless=${headless} &
wait

if [[ ${foxglove} == "true" ]]; then
  tmuxinator start -n foxglove -p tmuxinator/foxglove.yml &
  wait
fi

# Attach to tmux session ${drone_ns[@]}, window mission
tmux attach-session -t ${drone_ns[0]}:mission
