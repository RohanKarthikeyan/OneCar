import random
import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.error import InvalidAction

from objects_v3 import Car, Obstacle, Circle
import pygame

# Define game params.
STATE_W = 96
STATE_H = 96
VIDEO_W = 400
VIDEO_H = 600
FPS = 45  ## Frames per second

CAR_WIDTH = VIDEO_W*3//40
CAR_HEIGHT = VIDEO_H*3//40
OBSTACLE_WIDTH = VIDEO_W//20
OBSTACLE_HEIGHT = VIDEO_W//20

# Colors
WHITE = (255, 255, 255)
RED = (245, 30, 80)
PURPLE = (28, 46, 121)
TURQUOISE = (51, 204, 204)
BLUE_VIOLET = (150, 156, 230)
colors = [RED, TURQUOISE]


class OneCarEnv(gym.Env):
    """
    """
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "state_pixels"
        ],
        "render_fps": FPS
    }

    def __init__(self, render_mode=None, continuous=True):
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        pygame.init()
        random.seed(random.randint(0, 100))

        self.n_cars = 1
        self.lane_width = VIDEO_W // (2*self.n_cars)   # There are 2n_cars lanes in total
        self.screen_w = 2 * self.lane_width * self.n_cars
        self.screen_h = VIDEO_H

        self.continuous = continuous
        self.isopen = True
        self.clock = None

        self.all_sprites = pygame.sprite.Group()
        self.cars = pygame.sprite.Group()
        self.obstacles = pygame.sprite.Group()
        self.circles = pygame.sprite.Group()

        self.last_obj = []    # last_obj: last object introduced onto the field
        self.spawn_lane = []  # spawn_lane: lane in which the next object will be introduced

        for i in range(self.n_cars):
            car = Car(2*i+1, 2*i+2, colors[i])
            self.cars.add(car)
            self.all_sprites.add(car)
            car.set_lane(2 + i)

            self.last_obj.append(None)
            self.spawn_lane.append(random.randint(2*i+1, 2*i+2))

        self.score = 0
        self.prev_score = 0
        self.game_speed = 19

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8
            )

        if self.continuous:
            # -1 for left, 0 for nothing, 1 for right
            self.action_space = spaces.Box(low=-1, high=1)
        else:
            # 1 for left, 0 for nothing, 2 for right
            self.action_space = spaces.Discrete(3)

    def reset(self, *, seed = None, options=None):
        super().reset(seed=seed)

        self.all_sprites.empty()
        self.obstacles.empty()
        self.circles.empty()

        i = 0
        for car in self.cars:
            self.all_sprites.add(car)
            car.set_lane(2 + i)
            i += 1

        self.score = 0
        self.prev_score = 0
        self.game_speed = 19

        # Initialize last object and spawn lane for each car
        self.last_obj = [None] * self.n_cars
        self.spawn_lane = [random.randint(2*i+1, 2*i+2) for i in range(self.n_cars)]

        self.screen = pygame.display.set_mode((VIDEO_W, VIDEO_H))
        if self.render_mode == "human":
            self.render()
        if self.continuous:
            return self.step(action=[0.0])[0], {}  # Gym's LunarLander env
        else:
            return self.step(action=0)[0], {}

    def _hit_obstacle(self):
        # Check for collisions for either car
        for car in self.cars:
            if pygame.sprite.spritecollide(car, self.obstacles, False):
                return True

    def _update_score(self):
        # Check for collection of circles
        for car in self.cars:
            for circle in pygame.sprite.spritecollide(car, self.circles, False):
                self.score += 1
                circle.kill()

    def _has_missed_circles(self):
        # Check for missed circles
        for circle in self.circles:
            if circle.rect.y > self.screen_h - CAR_HEIGHT:
                return True

    def _spawn_objects(self):
        # Spawn new objects
        for i in range(self.n_cars):
            gap = random.choices([150, 250], [0.2, 0.8])[0]   # gap between objects

            if self.last_obj[i] == None or self.last_obj[i].rect.y > gap:
                self.spawn_lane[i] = random.choices([self.spawn_lane[i], 4*i-self.spawn_lane[i] + 3],
                                                    [0.2, 0.8])[0]   # lane of the next object
                obj = random.choices(['obstacle', 'circle'], [0.55, 0.45])[0]   # type of the next object   

                if obj == 'obstacle':
                    obstacle = Obstacle(self.spawn_lane[i], TURQUOISE)
                    self.all_sprites.add(obstacle)
                    self.obstacles.add(obstacle)
                    self.last_obj[i] = obstacle
                else:
                    circle = Circle(self.spawn_lane[i], RED)
                    self.all_sprites.add(circle)
                    self.circles.add(circle)
                    self.last_obj[i] = circle

    def step(self, action):
        """
        This is what happens at each time step in our environment:
        1. Decide if the car should move. If yes, in which direction?
            a. The amount to move is pre-determined by the game field's dimensions.
        2. Existing non-car objects move in the field.
        3. New non-car objects might be introduced onto the field.
        """
        # Check if the action is a valid action
        if self.continuous:
            self.cars.update(action[0], self.continuous)
        else:
            if not self.action_space.contains(action):
                raise InvalidAction(
                    f"You passed the invalid action `{action}`. " 
                    f"The supported action_space is `{self.action_space}`"  
                )
            self.cars.update(action, self.continuous)

        # Introduce new non-car objects
        self._spawn_objects()

        # Which direction does the car want to move? Then move.
        self.circles.update(self.game_speed)
        self.obstacles.update(self.game_speed)
        # self.cars.update(action)

        for obstacle in self.obstacles:  # Kill the obstacles we have dodged
            if obstacle.rect.y > self.screen_h - OBSTACLE_HEIGHT:
                obstacle.kill()

        step_reward = 0
        terminated = False
        truncated = False
        # Check for collisions or missed circles
        if(self._hit_obstacle() or self._has_missed_circles()):
            step_reward = -10  # New addition
            terminated = True

        # Update the score
        self.prev_score = self.score
        self._update_score()  # Check if circles have been collected
        step_reward = self.score - self.prev_score
        if self.score >= 200:
            truncated = True

        self.state = self._render("state_pixels")  # From CarRacing L561

        if self.render_mode == "human":
            self.render()
        return self.state, step_reward, terminated, truncated, {}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "Do specify the render_mode at initialization."
            )
            return
        else:
            return self._render(self.render_mode)

    def _render(self, mode):
        assert mode in self.metadata["render_modes"]

        if self.clock is None:
            self.clock = pygame.time.Clock()
        self.surf = pygame.Surface((VIDEO_W, VIDEO_H))

        # draw the canvas and objects
        self.surf.fill(PURPLE)  # Earlier self.screen
        self.all_sprites.draw(self.surf)  # Earlier self.screen

        # display score
        font = pygame.font.SysFont(None, 50)
        text = font.render(f"{self.score}", True, WHITE)
        self.surf.blit(text, (self.screen_w - 40, 10))  # Earlier self.screen

        # Draw lanes
        for i in range(1, 2*self.n_cars):
            pygame.draw.line(self.surf, BLUE_VIOLET,
                             (self.lane_width * i, 0),
                             (self.lane_width * i, VIDEO_H), 2)

        # self.surf = pygame.transform.flip(self.surf, False, True)
        # pygame.display.flip()  # Earlier uncommented

        # increase game speed with time proportional to score
        # self.game_speed += (self.score * 0.00001)

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
        # Crop the screen to remove the score
        cropped = pygame.Surface((self.screen_w, self.screen_h - 40))
        cropped.blit(screen, (0, 0), (0, 40, self.screen_w, self.screen_h))

        # Resize the screen
        scaled_screen = pygame.transform.smoothscale(cropped, size)
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(scaled_screen)), axes=(1, 0, 2)
        )

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            self.isopen = False
            pygame.quit()


if __name__ == "__main__":
    env = OneCarEnv(render_mode="human")
    action = np.array([0.0])  # following CarRacing

    def register_input():
        global game_quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_quit = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action[0] = -1.0
        elif keys[pygame.K_RIGHT]:
            action[0] = 1.0
        elif keys[pygame.K_ESCAPE]:
            game_quit = True

    game_quit = False
    while not game_quit:
        env.reset()
        total_score = 0.0

        while True:
            register_input()
            s, r, terminated, truncated, info = env.step(action)
            total_score += r
            if terminated or truncated or game_quit:
                break
        print(f"Game over! Your score was: {total_score}")
    env.close()