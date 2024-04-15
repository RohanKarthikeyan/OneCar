import pygame
import random
import numpy as np
# import gymnasium as gym
import matplotlib.pyplot as plt

# Define game params
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
STATE_W = 96
STATE_H = 96
LANE_WIDTH = SCREEN_WIDTH // 4     # There are 4 lanes in total
CAR_WIDTH = SCREEN_WIDTH*3//40
CAR_HEIGHT = SCREEN_HEIGHT*3//40
OBSTACLE_WIDTH = SCREEN_WIDTH//20
OBSTACLE_HEIGHT = SCREEN_WIDTH//20
GAME_SPEED = 7
FPS = 30

# rgb colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (245, 30, 80)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
PURPLE = (28, 46, 121)
BLUE_VIOLET = (150,156,230)
TURQUOISE = (51,204,204)

colours = [RED, TURQUOISE]


# Defining game objects
class Car(pygame.sprite.Sprite):
    def __init__(self, lane_start, lane_end, colour=RED):
        super().__init__()
        self.image = pygame.Surface((CAR_WIDTH, CAR_HEIGHT))
        self.image.fill(colour)
        self.rect = self.image.get_rect()
        self.lane_start = lane_start
        self.lane_end = lane_end
        self.rect.centerx = (2*lane_start - 1) * LANE_WIDTH // 2
        self.rect.bottom = SCREEN_HEIGHT - 10

    def get_lane(self):
        if (self.rect.centerx == (2*self.lane_end - 1) * LANE_WIDTH // 2):
            return self.lane_end
        elif (self.rect.centerx == (2*self.lane_start - 1) * LANE_WIDTH // 2):
            return self.lane_start
        else:
            return -1
        
    def set_lane(self, lane):
        self.rect.centerx = (2*lane - 1) * LANE_WIDTH // 2

    def update(self, actions):
        action = actions[(self.lane_start-1)//2]
        if action==1 or action==2:
            if self.get_lane() == self.lane_start:
                self.set_lane(self.lane_end)
            else:
                self.set_lane(self.lane_start)


class Obstacle(pygame.sprite.Sprite):
    def __init__(self, lane, colour = TURQUOISE):
        super().__init__()
        self.image = pygame.Surface((OBSTACLE_WIDTH, OBSTACLE_HEIGHT))
        self.image.fill(colour)
        self.rect = self.image.get_rect()
        self.rect.centerx = lane * LANE_WIDTH - LANE_WIDTH // 2
        self.rect.y = -OBSTACLE_HEIGHT

    def update(self, speed=GAME_SPEED):
        self.rect.y += speed

class Circle(pygame.sprite.Sprite):
    def __init__(self, lane, colour = TURQUOISE):
        super().__init__()
        self.image = pygame.Surface((OBSTACLE_WIDTH, OBSTACLE_HEIGHT))
        self.image.fill(PURPLE)
        self.rect = self.image.get_rect()
        self.rect.centerx = lane * LANE_WIDTH - LANE_WIDTH // 2
        self.rect.y = -OBSTACLE_HEIGHT

        # pygame.draw.polygon(self.image, TURQUOISE, [(0, OBSTACLE_HEIGHT), (OBSTACLE_WIDTH // 2, 0), (OBSTACLE_WIDTH, OBSTACLE_HEIGHT)])
        pygame.draw.circle(self.image, colour, (OBSTACLE_WIDTH // 2, OBSTACLE_HEIGHT // 2), OBSTACLE_HEIGHT // 2)

    def update(self, speed=GAME_SPEED):
        self.rect.y += speed

# class GameEnv(gym.Env):
class GameEnv():
    def __init__(self, n=2):
        pygame.init()
        random.seed(random.randint(0, 100))

        # Initialize game screen depending on number of cars
        self.n_cars = n
        self.screen_w = self.n_cars * LANE_WIDTH * 2
        self.screen_h = SCREEN_HEIGHT

        self.clock = pygame.time.Clock()

        # Initialize groups for storing game objects
        self.all_sprites = pygame.sprite.Group()
        self.cars = pygame.sprite.Group()
        self.obstacles = pygame.sprite.Group()
        self.circles = pygame.sprite.Group()

        self.last_obj = []    # last_obj: last object spawned for each car
        self.spawn_lane = []  # spawn_lane: lane in which the next object will be spawned for each car

        # Initialize cars
        for i in range(self.n_cars):
            car = Car(2*i+1, 2*i+2, colours[i])
            self.cars.add(car)
            self.all_sprites.add(car)
            car.set_lane(2 + i)

            # Initialize last object and spawn lane for each car
            self.last_obj[i] = None
            self.spawn_lane[i] = random.randint(2*i+1, 2*i+2)

        # Initialize gameplay parameters
        self.score = 0
        self.game_speed = GAME_SPEED

    def reset(self):
        # Reset game environment
        self.all_sprites.empty()
        self.obstacles.empty()
        self.circles.empty()

        i = 0
        for car in self.cars:
            self.all_sprites.add(car)
            car.set_lane(2 + i)
            i += 1

        self.score = 0
        self.game_speed = GAME_SPEED

        # Initialize last object and spawn lane for each car
        self.last_obj = [None] * self.n_cars
        self.spawn_lane = [random.randint(2*i+1, 2*i+2) for i in range(self.n_cars)]

        self.screen = pygame.display.set_mode((self.screen_w, self.screen_h))
        pygame.display.set_caption("2Cars Game")

        return self.render()

    def render(self):
        # draw the canvas and objects
        self.screen.fill(PURPLE)
        self.all_sprites.draw(self.screen)

        # display score - IMPORTANT: comment this part out when capturing frames for training
        font = pygame.font.SysFont(None, 50)
        text = font.render(f"{self.score}", True, WHITE)
        self.screen.blit(text, (self.screen_w - 40, 10))

        # draw lanes
        for i in range(1, 2*self.n_cars):
            pygame.draw.line(self.screen, BLUE_VIOLET, (LANE_WIDTH * i, 0), (LANE_WIDTH * i, SCREEN_HEIGHT), 2)
        pygame.display.flip()

        # increase game speed with time proportional to score
        self.game_speed += (self.score * 0.00001)

        # cap the frame rate
        self.clock.tick(FPS)

        # return the screen as an image array
        return self._create_image_array(self.screen, (STATE_W, STATE_H))
    
    def step(self, actions):
        # Spawn new objects
        self.spawn_objects()

        # Update all objects
        # self.all_sprites.update(self.game_speed)
        self.circles.update(self.game_speed)
        self.obstacles.update(self.game_speed)
        self.cars.update(actions)

        # Check for collisions or missed circles
        terminated = False
        if(self.has_collisions() or self.has_missed_circles()):
            terminated = True

        prev_score = self.score

        # Check for collection of circles
        self.update_score()
        truncated = self.score >= 300

        # Calculate 1-step reward
        step_reward = self.score - prev_score

        # Render the screen
        state = self.render()

        return state, step_reward, terminated, truncated, {}

    def has_collisions(self):
        # Check for collisions for either car
        for car in self.cars:
            if pygame.sprite.spritecollide(car, self.obstacles, False):
                return True
            
    def update_score(self):
        # Check for collection of circles
        for car in self.cars:
            for circle in pygame.sprite.spritecollide(car, self.circles, False):
                self.score += 1
                circle.kill()

    def has_missed_circles(self):
        # Check for missed circles
        for circle in self.circles:
            if circle.rect.y > SCREEN_HEIGHT - CAR_HEIGHT:
                return True
            
    def spawn_objects(self):
        # Spawn new objects
        for i in range (self.n_cars):
            # Choose a random spawning gap between 150 and 250
            gap = random.choices([150,250],[0.2,0.8])[0]

            if self.last_obj[i] == None or self.last_obj[i].rect.y > gap:
                # Randomly choose the lane in which the next object will be spawned
                self.spawn_lane[i] = random.choices([self.spawn_lane[i], 4*i-self.spawn_lane[i]+3],[0.2,0.8])[0]
                # Randomly choose the type of the next object
                obj = random.choices(['obstacle','circle'],[0.55,0.45])[0]   # type of the next object   
                if obj == 'obstacle':
                    obstacle = Obstacle(self.spawn_lane[i], colours[i])
                    self.all_sprites.add(obstacle)
                    self.obstacles.add(obstacle)
                    self.last_obj[i] = obstacle
                else:
                    circle = Circle(self.spawn_lane[i], colours[i])
                    self.all_sprites.add(circle)
                    self.circles.add(circle)
                    self.last_obj[i] = circle

    def _create_image_array(self, screen, size):
        # Crop the screen to remove the score
        cropped = pygame.Surface((self.screen_w, self.screen_h-40))
        cropped.blit(screen, (0, 0), (0, 40, self.screen_w, self.screen_h))
        
        # Resize the screen as per DQN requirements and return as image array
        scaled_screen = pygame.transform.smoothscale(cropped, size)
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(scaled_screen)), axes=(1, 0, 2)
        )
    
    def close(self):
        # Close the game environment
        if self.screen is not None:
            pygame.display.quit()
            self.isopen = False
            pygame.quit()



if __name__ == "__main__":

    env = GameEnv(n=2)

    frame_buffer = []

    # Start Game
    env.reset()
    terminated = False
    t=0

    # Game loop
    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        # Actions are taken once in 10 frames
        actions=[0,0]
        if t!=0 and t%10==0:
            # random policy for now
            actions = [random.randint(0,1), random.randint(0,1)]    # 0: stay, 1: left, 2: right
            t=-1

        s, r, terminated, truncated, info = env.step(actions)
        
        # Append the frame (game screen) to buffer
        frame_buffer.append(s)

        t+=1
        
    pygame.quit()

    # # Display the frames
    # for frame in frame_buffer:
    #     plt.imshow(frame)
    #     plt.pause(0.001)