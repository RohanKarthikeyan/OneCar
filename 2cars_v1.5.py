import pygame
import random
import numpy as np

# Define game params
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
STATE_W = 96
STATE_H = 96
LANE_WIDTH = SCREEN_WIDTH // 4     # There are 4 lanes in total
CAR_WIDTH = SCREEN_WIDTH*3//40
CAR_HEIGHT = SCREEN_HEIGHT*3//40
OBSTACLE_WIDTH = SCREEN_WIDTH/20
OBSTACLE_HEIGHT = SCREEN_WIDTH/20
GAME_SPEED = 7
FPS = 30

# rgb colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
PURPLE = (70,30,200)
BLUE_VIOLET = (150,156,230)
TURQUOISE = (64,224,208)


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

    def update(self, speed=GAME_SPEED):
        keys = pygame.key.get_pressed()
        if self.lane_start == 1 and self.lane_end == 2:
            if keys[pygame.K_a]:
                # if left key is pressed AND car is in right lane, move the car to the left lane
                if self.get_lane() == self.lane_end:
                    self.set_lane(self.lane_start)
            elif keys[pygame.K_d]:
                # if right key is pressed AND car is in left lane, move the car to the right lane
                if self.get_lane() == self.lane_start:
                    self.set_lane(self.lane_end)
        elif self.lane_start == 3 and self.lane_end == 4:
            if keys[pygame.K_LEFT]:
                # if left key is pressed AND car is in right lane, move the car to the left lane
                if self.get_lane() == self.lane_end:
                    self.set_lane(self.lane_start)
            elif keys[pygame.K_RIGHT]:
                # if right key is pressed AND car is in left lane, move the car to the right lane
                if self.get_lane() == self.lane_start:
                    self.set_lane(self.lane_end)


class Obstacle(pygame.sprite.Sprite):
    def __init__(self, lane, ):
        super().__init__()
        self.image = pygame.Surface((OBSTACLE_WIDTH, OBSTACLE_HEIGHT))
        self.image.fill(TURQUOISE)
        self.rect = self.image.get_rect()
        self.rect.centerx = lane * LANE_WIDTH + LANE_WIDTH // 2
        self.rect.y = -OBSTACLE_HEIGHT

    def update(self, speed=GAME_SPEED):
        self.rect.y += speed

class Circle(pygame.sprite.Sprite):
    def __init__(self, lane):
        super().__init__()
        self.image = pygame.Surface((OBSTACLE_WIDTH, OBSTACLE_HEIGHT))
        self.image.fill(PURPLE)
        self.rect = self.image.get_rect()
        self.rect.centerx = lane * LANE_WIDTH + LANE_WIDTH // 2
        self.rect.y = -OBSTACLE_HEIGHT

        # pygame.draw.polygon(self.image, TURQUOISE, [(0, OBSTACLE_HEIGHT), (OBSTACLE_WIDTH // 2, 0), (OBSTACLE_WIDTH, OBSTACLE_HEIGHT)])
        pygame.draw.circle(self.image, TURQUOISE, (OBSTACLE_WIDTH // 2, OBSTACLE_HEIGHT // 2), OBSTACLE_HEIGHT // 2)

    def update(self, speed=GAME_SPEED):
        self.rect.y += speed

class GameEnv():
    def __init__(self):
        pygame.init()
        random.seed(random.randint(0, 100))

        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("2Cars Game")
        self.clock = pygame.time.Clock()

        self.all_sprites = pygame.sprite.Group()
        self.cars = pygame.sprite.Group()
        self.obstacles = pygame.sprite.Group()
        self.circles = pygame.sprite.Group()

        self.score = 0
        self.game_speed = GAME_SPEED

        self.last_obj = None
        self.spawn_lane = random.randint(0, 1)

    def reset(self):
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

        self.last_obj = None
        self.spawn_lane = random.randint(0, 1)

    def render(self):
        # draw the canvas and objects
        self.screen.fill(PURPLE)
        self.all_sprites.draw(self.screen)

        # display score
        font = pygame.font.SysFont(None, 50)
        text = font.render(f"{self.score}", True, WHITE)
        self.screen.blit(text, (SCREEN_WIDTH - 40, 10))

        # draw lanes
        for i in range(1, 4):
            pygame.draw.line(self.screen, BLUE_VIOLET, (LANE_WIDTH * i, 0), (LANE_WIDTH * i, SCREEN_HEIGHT), 2)
        pygame.display.flip()

        # increase game speed with time proportional to score
        self.game_speed += (self.score * 0.0001)

        # cap the frame rate
        self.clock.tick(FPS)

        return self._create_image_array(self.screen, (STATE_W, STATE_H))

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
        gap = random.choices([150,250],[0.2,0.8])[0]   # gap between objects
        self.spawn_lane = random.choices([self.spawn_lane,1-self.spawn_lane],[0.2,0.8])[0]   # lane of the next object
        obj = random.choices(['obstacle','circle'],[0.5,0.5])[0]   # type of the next object
        if self.last_obj == None or self.last_obj.rect.y > gap:
            if obj == 'obstacle':
                obstacle = Obstacle(self.spawn_lane)
                self.all_sprites.add(obstacle)
                self.obstacles.add(obstacle)
                self.last_obj = obstacle
            else:
                circle = Circle(self.spawn_lane)
                self.all_sprites.add(circle)
                self.circles.add(circle)
                self.last_obj = circle

    def _create_image_array(self, screen, size):
        scaled_screen = pygame.transform.smoothscale(screen, size)
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(scaled_screen)), axes=(1, 0, 2)
        )
    
    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            self.isopen = False
            pygame.quit()



if __name__ == "__main__":

    env = GameEnv()

    car1 = Car(1, 2, TURQUOISE)
    car2 = Car(3, 4, RED)
    env.cars.add(car1)
    env.cars.add(car2)
    env.all_sprites.add(car1)
    env.all_sprites.add(car2)

    # Game loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        # Spawn new objects
        env.spawn_objects()

        # Update all objects
        env.all_sprites.update(env.game_speed)

        # Check for collisions or missed circles
        if(env.has_collisions() or env.has_missed_circles()):
            running = False

        # Check for collection of circles
        env.update_score()

        # Render the screen
        env.render()

    pygame.quit()
