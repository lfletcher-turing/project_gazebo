import argparse
import threading
import numpy as np
from typing import List
import rclpy
from rclpy.node import Node
from as2_python_api.drone_interface_teleop import DroneInterfaceTeleop
from as2_python_api.behavior_actions.behavior_handler import BehaviorHandler
from as2_msgs.msg import BehaviorStatus
from vision_msgs.msg import Detection3DArray
import csv
from geometry_msgs.msg import PoseStamped
from scipy.spatial.distance import pdist
from std_msgs.msg import Float64

dim = 5.0
height = 10.0

path = [
        [-dim, dim, height],
        [-dim, -dim, height],
        [dim, -dim, height],
        [dim, dim, height]
    ]

class VisionPositionNode(Node):
    def __init__(self, drone):
        super().__init__(f'{drone.namespace}_vision_position')
        self.drone = drone
        self.subscription = self.create_subscription(Detection3DArray, f'{drone.namespace}/filtered_detections', self.relative_pos_callback, 10)
        self.relative_positions = None


    def relative_pos_callback(self,  msg):
        # print("Callback called")
        self.drone.relative_positions = np.zeros((len(msg.detections), 3))
        for i, detection in enumerate(msg.detections):
            self.drone.relative_positions[i] = np.array([detection.bbox.center.position.x, detection.bbox.center.position.y, detection.bbox.center.position.z])
        # print(f"Relative positions: {self.drone.relative_positions}")


class FlockingSwarm:
    def __init__(self, drones_ns: List[str], config: dict, migration_path: List[List[float]] = None, max_migration_distance = 10, use_gps = False, bagging = True):
        self.drones: dict[int, FlockingDrone] = {}
        self.config = config
        self.migration_path = migration_path
        self.max_migration_distance = max_migration_distance
        self.random_waypoint = None
        self.use_gps = use_gps
        self.bagging = bagging
        self.migration_index = 0
        for index, name in enumerate(drones_ns):
            self.drones[index] = FlockingDrone(name, index, config=config, use_gps=use_gps)

        # if no migration path, generate random waypoint
        if not migration_path:
            self.random_migration = True
            self.generate_random_waypoint()
        else:
            self.random_migration = False
            

        # give each drone a reference to the other drones
        for drone in self.drones.values():
            drone.other_drones = [d for d in self.drones.values() if d.my_id != drone.my_id]
            if self.migration_path:
                drone.set_migration_path(self.migration_path)



    def get_ready(self):
        for drone in self.drones.values():
            drone.arm()
            drone.offboard()

    def offboard(self):
        for drone in self.drones:
            drone.offboard()

    def takeoff(self, altitude: float):
        for drone in self.drones.values():
            drone.do_behavior("takeoff", altitude, 5, False)
        self.wait()

    def shutdown(self):
        for drone in self.drones.values():
            drone.shutdown()

    def wait(self):
        all_finished = False
        while not all_finished:
            all_finished = True
            for drone in self.drones.values():
                all_finished = all_finished and drone.goal_reached()

    def land(self):
        for drone in self.drones.values():
            drone.do_behavior("land", 0.4, False)
        self.wait()

    def flock(self):
        print("Flocking")
        while True:
            if self.random_migration:
                self.update_random_waypoint()
            else:
                self.update_migration_index()
                if self.migration_index == len(self.migration_path):
                    # go back to the first migration point
                    self.migration_index = 0 
            for drone in self.drones.values():
                drone.set_migration_index(self.migration_index)
                drone.flock()
   

    def distance_stats(self, positions: np.ndarray):
        distances = pdist(positions)
        self.distance_mean_pub.publish(Float64(distances.mean()))

        

          
    def update_migration_index(self):
        if self.migration_path:
            current_goal = np.array(self.migration_path[self.migration_index])

            swarm_pos = np.mean([d.position for d in self.drones.values()], axis=0)

            distance = np.linalg.norm(swarm_pos - current_goal)

            if distance < self.config['migration_threshold']:
                self.migration_index = (self.migration_index + 1) % len(self.migration_path)
                # print(f"Switching to migration index {self.migration_index}")

    def update_random_waypoint(self):
        if self.random_waypoint is None:
            self.generate_random_waypoint()
        else:
            swarm_pos = np.mean([d.position for d in self.drones.values()], axis=0)
            distance = np.linalg.norm(swarm_pos - self.random_waypoint)

            if distance < self.config['migration_threshold']:
                self.generate_random_waypoint()

    def generate_random_waypoint(self):
        self.random_waypoint = [
            np.random.uniform(-self.max_migration_distance, self.max_migration_distance),
            np.random.uniform(-self.max_migration_distance, self.max_migration_distance),
            self.config['altitude_setpoint']
        ]
        # print(f"New random waypoint: {self.random_waypoint}")
        for drone in self.drones.values():
            drone.set_target_position(self.random_waypoint)


class FlockingDrone(DroneInterfaceTeleop):

    def __init__(self, namespace: str, my_id: int, config: dict, use_gps = False, bagging = True):
        super().__init__(namespace, use_sim_time=True)
        self.my_id = my_id
        self.config = config
        self.other_drones = None
        self.use_gps = use_gps
        self.target_position = None
        self.migration_path = None
        self.bagging = bagging

        self.last_command = np.zeros(3)
        self.detections = []
        self.poses = {}

        if self.bagging:
            self.migration_goal_pub = self.create_publisher(PoseStamped, f'/{namespace}/migration_waypoint', 10)

    def set_target_position(self, target_position: List[float]):
        self.target_position = target_position

    def flock(self):
        # Get command from reynolds
        command = self.get_command_reynolds()
        
        # Low-pass filter command
        command = self.smooth_command(command)

        # Scale by gain and clip to max speed 
        command = self.process_command(command)

        # Altitude control
        command = self.add_altitude_control(command)

        # Add migration 
        command = self.add_migration(command)
        
        # Send velocity command
        self.send_velocity_command(command)

    def get_command_reynolds(self):
        # Use either visual detections or other drones poses
        positions_rel = self.get_relative_positions()

        # if positions_rel is empty, return zero command
        if positions_rel is None or not positions_rel.any():
            return np.zeros(3)
        
        # Separation
        sep = self.separation(positions_rel)
        
        # Cohesion 
        coh = self.cohesion(positions_rel)
        velocities_rel = np.zeros((len(self.other_drones), 3))
        # Alignment
        align = self.alignment(velocities_rel)

        # Combine behaviors
        command = (
            -self.config['separation_gain'] * sep.mean(axis=0) +
            self.config['cohesion_gain'] * coh + 
            self.config['alignment_gain'] * align
        )

        return command
    
    # def separation(self, positions_rel: List[np.ndarray]) -> np.ndarray:
    #     sep = np.zeros(3)
    #     for pos in positions_rel:
    #         dist = np.linalg.norm(pos)
    #         if 0 < dist < self.config['separation_dist']:
    #             sep -= pos / dist
    #     return sep

    def separation(self, positions_rel: List[np.ndarray]) -> np.ndarray:
        dist_inv = np.zeros(3)
        if positions_rel is not None:
            if positions_rel.any():
                positions = np.array(positions_rel)
                distances = np.linalg.norm(positions, axis=1)
                safe_distances = np.where(distances > 0.01, distances, np.inf)
                # print(f"safe_distances: {safe_distances}")

                dist_inv = positions / safe_distances[:, np.newaxis] ** 2

        # print(f"separation: {dist_inv}")

        return dist_inv


    def cohesion(self, positions_rel: List[np.ndarray]) -> np.ndarray:
        # print(f"positions_rel: {positions_rel}")
        coh = np.zeros(3)
        if positions_rel is not None:
            if positions_rel.any():
                coh = np.mean(positions_rel, axis=0) 
                # print(f"cohesio
                # n: {coh}")
            # print(f"cohesion:
            #  {coh}")
        return coh
    
    def alignment(self, velocities_rel: List[np.ndarray]) -> np.ndarray:
        align = np.zeros(3)
        if velocities_rel.any():
            align = np.mean(velocities_rel, axis=0)
        # print(f"alignment: {align}")
        return align

    def smooth_command(self, command: np.ndarray) -> np.ndarray:
        alpha = self.config['smoothing_factor']
        smoothed = alpha * command + (1 - alpha) * self.last_command
        self.last_command = smoothed
        return smoothed

    def process_command(self, command: np.ndarray) -> np.ndarray:
        # Scale by gain
        command *= self.config['command_gain']
        
        # Clip to max speed
        max_speed = self.config['max_speed']
        if np.linalg.norm(command) > max_speed:
            command = command / np.linalg.norm(command) * max_speed

        return command

    def add_altitude_control(self, command: np.ndarray) -> np.ndarray:
        if self.config['use_altitude']:
            altitude = self.get_altitude()
            alt_error = self.config['altitude_setpoint'] - altitude
            command[2] = self.config['altitude_gain'] * alt_error
        return command
            
    def add_migration(self, command: np.ndarray) -> np.ndarray:
        if self.migration_path:
            current_goal =np.array(self.migration_path[self.migration_index])
        else:
            current_goal = np.array(self.target_position)
        
        # publish the current goal
        msg = PoseStamped()
        msg.pose.position.x = current_goal[0]
        msg.pose.position.y = current_goal[1]
        msg.pose.position.z = current_goal[2]
        self.migration_goal_pub.publish(msg) 

        # print(f"Cur /rent go al: {current_goal}")
        current_position = np.array(self.position)
        # print(f" /Current position: {current_position}")
        direction = current_goal - current_position
        distance = np.linalg.norm(direction)
        direction = direction / distance
        command += self.config['migration_gain'] * direction
                # convert to list

        return command
        

    def set_migration_path(self, path: List[List[float]]):
        self.migration_path = path
        self.migration_index = 0

    def set_migration_index(self, index: int):
        self.migration_index = index

    def send_velocity_command(self, 
                              command: np.ndarray = np.zeros(3),  
                              frame_id: str = 'earth',
                              yaw_speed: float = 0.0):
        
        # convert to list
        command = command.tolist()
        
        # print the command
        # print(f"Sending command {command} to drone {self.my_id}")
        
        self.motion_ref_handler.speed.send_speed_command_with_yaw_speed(command,  frame_id, yaw_speed)


    def get_relative_positions(self) -> List[np.ndarray]:

        # either from visual detections or pose estimates
        positions_rel = []
        positions_rel_gps = []

        my_gps = np.array(self.position)

        if self.use_gps:
            for other_drone in self.other_drones:
                other_gps = np.array(other_drone.position)
                
                rel_pos = other_gps - my_gps

                # print the relative position of the other drones
                # print(f"Relative position of drone {other_drone.my_id} is {rel_pos}")

                # convert back to list

                rel_pos = rel_pos.tolist()

                positions_rel_gps.append(rel_pos)
        else:
            positions_rel = self.relative_positions
        
        # if self.logger:
        #     self.logger.log_data(self.get_clock().now().nanoseconds, self.namespace, positions_rel, positions_rel_gps)

        return np.array(positions_rel_gps) if self.use_gps else positions_rel
    

    def get_altitude(self) -> float:
        
        return self.position[2]
    
    def do_behavior(self, behavior: str, *args):
        self.current_behavior = getattr(self, behavior)
        self.current_behavior(*args)

    def goal_reached(self) -> bool:
        """Check if current behavior has finished"""
        if not self.current_behavior:
            return False

        if self.current_behavior.status == BehaviorStatus.IDLE:
            return True
        return False
    


if __name__ == '__main__':

    dim = 5.0
    height = 10.0

    path_square = [
        [-dim, dim, height],
        [-dim, -dim, height],
        [dim, -dim, height],
        [dim, dim, height]
        ]
    
    path_line = [
        [-3, 0, height],
        [3, 0, height]
    ]


    parser = argparse.ArgumentParser()
    parser.add_argument('--use_gps', action='store_true', help='Use GPS for flocking')
    parser.add_argument('--bagging', action='store_true', help='Use bagging')

    args = parser.parse_args()

    print(f"Using GPS: {args.use_gps}")

    drones_ns = ['drone0', 'drone1', 'drone2']
    config = {
        'separation_gain': 7.0,
        'cohesion_gain': 1.0,
        'alignment_gain': 1.0,
        'separation_dist': 2.0,
        'command_gain': 1.0,
        'max_speed': 0.5,
        'smoothing_factor': 1.0,
        'use_altitude': True,
        'altitude_setpoint': 10.0,
        'altitude_gain': 0.5,
        'migration_threshold': 2.0,
        'migration_gain': 1.0
    }

    rclpy.init()

    swarm = FlockingSwarm(drones_ns, config, use_gps=args.use_gps, bagging=args.bagging, migration_path=path_line)

    if not args.use_gps:
        vision_nodes = [VisionPositionNode(drone) for drone in swarm.drones.values()]

        executor = rclpy.executors.MultiThreadedExecutor(num_threads=len(vision_nodes))
        for node in vision_nodes:
            executor.add_node(node)
        def spin_in_thread(executor):
            executor.spin()

        executor_thread = threading.Thread(target=spin_in_thread, args=(executor,))
        executor_thread.start()

    swarm.get_ready()
    swarm.takeoff(10.0)
    swarm.flock()

    swarm.land()
    swarm.shutdown()
    rclpy.shutdown()
    if not args.use_gps:
        executor_thread.join()
        for node in vision_nodes:
            node.destroy_node()

