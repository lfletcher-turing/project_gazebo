import numpy as np
from typing import List
import rclpy
from as2_python_api.drone_interface_teleop import DroneInterfaceTeleop
from as2_python_api.behavior_actions.behavior_handler import BehaviorHandler
from as2_msgs.msg import BehaviorStatus

dim = 5.0
height = 5.0

path = [
        [-dim, dim, height],
        [-dim, -dim, height],
        [dim, -dim, height],
        [dim, dim, height]
    ]

class FlockingSwarm:
    def __init__(self, drones_ns: List[str], config: dict, migration_path: List[List[float]] = path):
        self.drones: dict[int, FlockingDrone] = {}
        self.config = config
        self.migration_path = migration_path
        self.migration_index = 0
        for index, name in enumerate(drones_ns):
            self.drones[index] = FlockingDrone(name, index, config)

        # give each drone a reference to the other drones
        for drone in self.drones.values():
            drone.other_drones = [d for d in self.drones.values() if d.my_id != drone.my_id]
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
            drone.do_behavior("takeoff", altitude, 0.7, False)
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
        while True:
            self.update_migration_index()
            for drone in self.drones.values():
                drone.set_migration_index(self.migration_index)
                drone.flock()
    
    def update_migration_index(self):
        if self.migration_path:
            current_goal = np.array(self.migration_path[self.migration_index])

            swarm_pos = np.mean([d.position for d in self.drones.values()], axis=0)

            distance = np.linalg.norm(swarm_pos - current_goal)

            if distance < self.config['migration_threshold']:
                self.migration_index = (self.migration_index + 1) % len(self.migration_path)
                print(f"Switching to migration index {self.migration_index}")



class FlockingDrone(DroneInterfaceTeleop):

    def __init__(self, namespace: str, my_id: int, config: dict):
        super().__init__(namespace)
        self.my_id = my_id
        self.config = config
        self.other_drones = None
        
        self.last_command = np.zeros(3)
        self.detections = []
        self.poses = {}

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
        
        # Separation
        sep = self.separation(positions_rel)
        
        # Cohesion 
        coh = self.cohesion(positions_rel)
        
        # Alignment
        align = self.alignment(positions_rel)

        # Combine behaviors
        command = (
            self.config['separation_gain'] * sep +
            self.config['cohesion_gain'] * coh + 
            self.config['alignment_gain'] * align
        )

        return command
    
    def separation(self, positions_rel: List[np.ndarray]) -> np.ndarray:
        sep = np.zeros(3)
        for pos in positions_rel:
            dist = np.linalg.norm(pos)
            if 0 < dist < self.config['separation_dist']:
                sep -= pos / dist
        return sep

    def cohesion(self, positions_rel: List[np.ndarray]) -> np.ndarray:
        coh = np.zeros(3)
        if positions_rel:
            coh = np.mean(positions_rel, axis=0) 
        return coh
    
    def alignment(self, velocities_rel: List[np.ndarray]) -> np.ndarray:
        align = np.zeros(3)
        if velocities_rel:
            align = np.mean(velocities_rel, axis=0)
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
            current_position = np.array(self.position)
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

        my_gps = np.array(self.position)

        for other_drone in self.other_drones:
            other_gps = np.array(other_drone.position)
            
            rel_pos = other_gps - my_gps

            # print the relative position of the other drones
            # print(f"Relative position of drone {other_drone.my_id} is {rel_pos}")

            # convert back to list

            rel_pos = rel_pos.tolist()

            positions_rel.append(rel_pos)

        return positions_rel


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
    drones_ns = ['drone0', 'drone1', 'drone2']
    config = {
        'separation_gain': 5.0,
        'cohesion_gain': 1.0,
        'alignment_gain': 1.0,
        'separation_dist': 2.0,
        'command_gain': 1.0,
        'max_speed': 0.5,
        'smoothing_factor': 1.0,
        'use_altitude': True,
        'altitude_setpoint': 5.0,
        'altitude_gain': 0.5,
        'migration_threshold': 2.0,
        'migration_gain': 1.0
    }

    rclpy.init()

    swarm = FlockingSwarm(drones_ns, config)

    swarm.get_ready()
    swarm.takeoff(1.0)
    swarm.flock()
