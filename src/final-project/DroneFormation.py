import numpy as np
from DroneUtils import get_R_from_normal_vector

class DroneFormationAbstract():
    def __init__(self, n_drones, *args, **kwargs):
        self.n_drones = n_drones
    def get_relative_positions(self, num_drones):
        raise NotImplementedError("Subclasses must implement this method")

class CircleFormation(DroneFormationAbstract):
    def __init__(self, n_drones, radius=2.0,*args, **kwargs):
        super().__init__(n_drones, *args, **kwargs)
        self.radius = radius
    def get_relative_positions(self, num_drones, normal=[0, 0, 1], phase=0):
        
        R = get_R_from_normal_vector(normal)

        positions = []
        for i in range(num_drones):
            angle = 2 * np.pi * i / num_drones + phase
            x = self.radius * np.cos(angle)
            y = self.radius * np.sin(angle)
            pos_vector = np.matrix([[x], [y], [0]]) # 3 x 1
            rotated_pos_vector = R @ pos_vector # 3 x 1
            positions.append(rotated_pos_vector)

        return positions

class DroneFormation:
    def __new__(cls, n_drones, formation_type, *args, **kwargs):
        if formation_type == "circle":
            return CircleFormation(n_drones, *args, **kwargs)
        else:
            raise ValueError(f"Unknown formation_type: {formation_type}")