import random
from typing import Optional

import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.error import InvalidAction

from objects import Car, Obstacle, Circle
import pygame

STATE_W = 96
STATE_H = 96
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800
FPS = 30  ## Frames per second
OBJ_SPEED = 7  ## Speed by which non-car objects move

class OneCar(gym.Env):
    """
    ## Description

    ## Action Space
    The action shape is `(1,)` in the range `{0, 2}` indicating
    which direction to move the car in:
    - 0: do nothing
    - 1: go left
    - 2: go right

    ## Observation Space
    A top-down 96x96 greyscale image of the car and race track.

    ## Rewards
    The reward is +1 for every circle collected.

    ## Starting state
    The car starts in the center of the left lane.

    ## Episode Termination
    The episode ends if any one of the following happens:

    1. Termination: We miss collecting a circle.
    2. Termination: We collide with an obstacle.
    3. Truncation: The score is greater than 300.
    """
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "state_pixels"
        ],
        "render_fps": FPS
    }

    def __init__(self, render_mode = None):
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.screen: Optional[pygame.Surface] = None
        self.clock = None
        self.isopen = True
        self.score = 0

        self.objects = []
        # Store the last spawned triangles or circles
        # 'left'/'right' -> The last object spawned in the left/right lane
        # 'triangle'/'circle' -> The last triangle/circle spawned
        self.last_objects = {'left': 0, 'right': 0, 'obstacle': 0, 'circle': 0}

        # Defines where we move to when we take an action
        self.left_position = np.array(
            [int(self.observation_shape[0] * 0.9),
            int(self.observation_shape[1]/4) - int(self.car_w/2)]
            )
        self.right_position = np.array(
            [int(self.observation_shape[0] * 0.9),
            int(3*self.observation_shape[1]/4) - int(self.car_w/2)]
            )
        # Will be useful for spawning circles and triangles
        self.width_dict = {'left': self.left_position[1],
                           'right': self.right_position[1]}

        self.car = Car(self.left_position[0], self.left_position[1], 'left')

        self.observation_shape = (STATE_H, STATE_W)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=self.observation_shape, dtype=np.uint8
            )

        self.action_space = spaces.Discrete(3)  # Do nothing, left, right

        # Defining bounds on where the car can be placed
        # I believe x_min should be obs_shape[0] and y_min, obs_shape[1]
        self.x_min = int(self.observation_shape[1] * 0.05)
        self.y_min = int(self.observation_shape[0] * 0.05)
        self.x_max = int(self.observation_shape[1] * 0.95)
        self.y_max = int(self.observation_shape[0] * 0.95)

        # Defining structure of the car
        self.car_w = int(self.observation_shape[1] * 0.10)
        self.car_h = int(self.observation_shape[0] * 0.10)

        # Defining structure of the circles / triangles
        self.objects_dim = int(self.observation_shape[1] * 0.1)

    def reset(self, *, seed = None):
        super().reset(seed=seed)

        # Reset the car back to its position
        self.car = Car(self.left_position[0], self.left_position[1], 'left')
        self.score = 0
        self.objects = [self.car]
        self.last_objects = {'left': 0, 'right': 0, 'obstacle': 0, 'circle': 0}

        if self.render_mode == "human":
            self.render()
        return self.step(action=0)[0], {}  # Gym's LunarLander env

    def _spawn_object(self, obstacle=True):
        key = 'obstacle' if obstacle else 'circle'

        # Spawn an element with 50% chance
        if random.random() < 0.5:
            if self.last_objects[key] == 0:  # At the start of game
                new_lane = random.choices(['left', 'right'], [0.5, 0.5])[0]
        else:  # We want to spawn an element at the opposite side of where it came last
            last_obj_lane = self.last_objects[key].lane
            other_lane = 'right' if last_obj_lane == 'left' else 'left'
            new_lane = random.choices([last_obj_lane, other_lane], [0.1, 0.9])[0]

        """
        Condition 1:
            Object spawned should be atleast two car lengths (height)
            away from the last object spawned in that lane
        Condition 2:
            If the last object spawned is on the other lane, there must
            be atleast 3 car lengths (heights) distance between them
        If it satisfies both the conditions, it is added to self.elements.
        """
        last = self.last_objects[new_lane]  # last object in that lane
        last_obj = self.last_objects[key]  # last object of that type
        ### Here, y_min should be x_min (line 50, Initialization)
        min_inter_obj_dist = self.y_min + self.objects_dim + 2 * self.car_h

        """
        last.position[0] > min_inter_obj_dist: the last object spawned
            in that side has travelled at least `min_inter_obj_dist` distance
            down the screen from the top, and so we can spawn a new object; so,
            distances between consecutive objects will be at least
            `min_inter_obj_dist` (a lower bound)
        """
        # Check: Has last element traveled a safe distance downward?
        if last == 0 or last.position[0] > min_inter_obj_dist:
            # last_obj == 0 can come only when last == 0 (at start of game)
            # While the game is in progress, the other conditions are checked:
            # Either the last object has been spawned in the lane we chose (cf. lines 8, 12)
            # or it is the other lane. 
            # If it's on the other lane, we add `self.car_h` so that the car has
            # enough space to go around the objects. Imagine two obstacles on both
            # lanes at the same level!
            if last_obj == 0 or last_obj.lane == new_lane or (
                last_obj.lane != new_lane and last_obj.position[0] > min_inter_obj_dist + self.car_h
                ):
                # y_min must be x_min
                if obstacle:
                    elem = Obstacle(self.y_min + self.objects_dim,
                                    self.width_dict[new_lane], new_lane)
                else:
                    elem = Circle(self.y_min + self.objects_dim,
                                  self.width_dict[new_lane], new_lane)

                self.objects.append(elem)
                self.last_objects[new_lane] = elem
                self.last_objects[key] = elem

    def _has_collided(self, car, object):
        side = car.lane == object.lane

        car_x, obj_x = car.position[0], object.position[0]
        x_col = car_x - self.car_h <= obj_x <= car_x
        return x_col and side

    def step(self, action):
        """
        This is what happens at each time step in our environment:
        1. Decide if the car should move. If yes, in which direction?
            a. The amount to move is pre-determined by the game field's dimensions.
        2. Existing non-car objects move in the field.
        3. New non-car objects might be introduced onto the field.
        """
        # Check if the action is a valid action
        if not self.action_space.contains(action):
            raise InvalidAction(
                f"You passed the invalid action `{action}`. " 
                f"The supported action_space is `{self.action_space}`"  
            )

        # Step 1: Which direction does the car want to move? Then move.
        # Map actions to the step size we should take on the grid
        # The car is stationary at the (near)-bottom of the screen
        # So, there is no change in the x-coordinate position
        self._action_to_dist = {
            0: np.array([0, 0]),  # Do nothing
            1: np.array([0, -self.observation_shape[1]//2]),  # Left
            2: np.array([0, self.observation_shape[1]//2])  # Right
            }
        move_dist = self._action_to_dist[action]

        self._lane_dict = {1: 'Left', 2: 'Right'}
        if action != self.car.lane:
            self.car.step(move_dist, self._lane_dict[action])

        step_reward = 0
        terminated = False
        # Increase game speed with time proportional to score
        OBJ_SPEED += (self.score * 0.15)
        truncated = False
        for obj in self.objects:
            # Step 2a: Check if the step taken is valid
            if self._has_collided(self.car, obj) and isinstance(obj, Obstacle):
                terminated = True
            if self._has_collided(self.car, obj) and isinstance(obj, Circle):
                self.score += 1
                step_reward = 1
                self.objects.remove(obj)

            # Step 2b: Move the existing non-car objects
            if isinstance(obj, Obstacle):
                # Should be x_max
                if obj.position[0] >= self.y_max:
                    self.objects.remove(obj)  # Remove from view if we have dodged it
                else:
                    obj.step(np.array([OBJ_SPEED, 0]), obj.lane)

            if isinstance(obj, Circle):
                # Should be x_max
                if obj.get_position()[0] >= self.y_max:
                    terminated = True  # You missed a circle - game over...
                else:
                    obj.step(np.array([OBJ_SPEED, 0]), obj.lane)

        # Step 3: Introduce new non-car objects
        spawn_first = random.choices(['obstacle', 'circle'], [0.5, 0.5])[0]
        if spawn_first == 'obstacle':
            self._spawn_object(obstacle=True)  # which itself is a random action
            self._spawn_object()
        else:
            self._spawn_object()
            self._spawn_object(obstacle=True)

        self.state = self._render("state_pixels")  # From CarRacing L561

        if self.render_mode == "human":
            self.render()
        # Below is Gym's code for LunarLander
        truncated = self.score >= 300
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return self.state, step_reward, terminated, truncated, {}

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "Do specify the render_mode at initialization."
            )
            return
        else:
            return self._render(self.render_mode)

    def _render(self, mode):
        assert mode in self.metadata["render_modes"]

        pygame.font.init()
        if self.screen is None and mode == "human":
            pygame.init()
            pygame.display.init()
            # pygame.display.set_caption("One Car")
            self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((WINDOW_W, WINDOW_H))

        #########################
        ## Render the objects on self.surf
        # Define a scale factor (?)
        #########################

        ## Don't know if this is needed or not
        self.surf = pygame.transform.flip(self.surf, False, True)

        #########################
        ## Show the game score
        font = pygame.font.Font(pygame.font.get_default_font(), 42)
        text = font.render(f"{self.score}", True, (255, 255, 255))
        self.surf.blit(text, (WINDOW_W - 40, 10))
        #########################

        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            assert self.screen is not None

            self.screen.fill(0)
            self.screen.blit(self.surf, (0, 0))
            pygame.display.flip()
        elif mode == "rgb_array":
            return self._create_image_array(self.surf, (VIDEO_W, VIDEO_H))
        elif mode == "state_pixels":
            return self._create_image_array(self.surf, (STATE_W, STATE_H))
        else:
            return self.isopen

    def _create_image_array(self, screen, size):
        scaled_screen = pygame.transform.smoothscale(screen, size)
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(scaled_screen)), axes=(1, 0)
        )

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            self.isopen = False
            pygame.quit()


if __name__ == "__main__":
    env = OneCar(render_mode="human")

    episode_over = False
    total_score = 0
    while not episode_over:
        env.reset()
        while True:
            action = env.action_space.sample()
            s, r, terminated, truncated, info = env.step(action)
            total_score += r
            episode_over = terminated or truncated
    env.close()
    print(f"Game over! Your score was: {total_score}")