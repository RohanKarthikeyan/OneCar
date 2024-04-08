import numpy as np

class Car:
    """
    A basic Car object.
    """
    def __init__(self, init_x, init_y, init_lane):
        """
        Initialize the car's position on the grid.
        """
        self.position = np.array([init_x, init_y], dtype=int)
        self.lane = init_lane

    def step(self, delta_pos, lane):
        self.position += delta_pos
        self.lane = lane


class Obstacle:
    """
    A basic Obstacle object.
    """
    def __init__(self, init_x, init_y, init_lane):
        """
        Initialize the Obstacle's position on the grid.
        """
        self.position = np.array([init_x, init_y], dtype=int)
        self.lane = init_lane

    def step(self, delta_pos, lane):
        self.position += delta_pos
        self.lane = lane


class Circle:
    """
    A basic Circle object.
    """
    def __init__(self, init_x, init_y, init_lane):
        """
        Initialize the Circle's position on the grid.
        """
        self.position = np.array([init_x, init_y], dtype=int)
        self.lane = init_lane

    def step(self, delta_pos, lane):
        self.position += delta_pos
        self.lane = lane