import pygame

# Define game params.
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600

LANE_WIDTH = SCREEN_WIDTH // 2     # There are 2 lanes in total
CAR_WIDTH = SCREEN_WIDTH*3//40
CAR_HEIGHT = SCREEN_HEIGHT*3//40

OBSTACLE_WIDTH = SCREEN_WIDTH//20
OBSTACLE_HEIGHT = SCREEN_WIDTH//20
OBJ_SPEED = 7

# Colors
RED = (245, 30, 80)
PURPLE = (28, 46, 121)
TURQUOISE = (51, 204, 204)

# Defining the objects
class Car(pygame.sprite.Sprite):
    """
    A basic Car object.
    """
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

    def update(self, action):
        # 0: do nothing, 1: left, 2: right
        # action = actions[(self.lane_start-1)//2]
        if action==2 and self.get_lane() == self.lane_start:
            self.set_lane(self.lane_end)
        elif action==1 and self.get_lane() == self.lane_end:
            self.set_lane(self.lane_start)

class Obstacle(pygame.sprite.Sprite):
    """
    The obstacle object.
    """
    def __init__(self, lane, colour = TURQUOISE):
        super().__init__()
        self.image = pygame.Surface((OBSTACLE_WIDTH, OBSTACLE_HEIGHT))
        self.image.fill(colour)
        self.rect = self.image.get_rect()
        self.rect.centerx = lane * LANE_WIDTH - LANE_WIDTH // 2
        self.rect.y = -OBSTACLE_HEIGHT

    def update(self, speed=OBJ_SPEED):
        self.rect.y += speed

class Circle(pygame.sprite.Sprite):
    """
    The circle object.
    """
    def __init__(self, lane, colour = TURQUOISE):
        super().__init__()
        self.image = pygame.Surface((OBSTACLE_WIDTH, OBSTACLE_HEIGHT))
        self.image.fill(PURPLE)
        self.rect = self.image.get_rect()
        self.rect.centerx = lane * LANE_WIDTH - LANE_WIDTH // 2
        self.rect.y = -OBSTACLE_HEIGHT

        pygame.draw.circle(self.image, colour,
                           (OBSTACLE_WIDTH // 2, OBSTACLE_HEIGHT // 2),
                            OBSTACLE_HEIGHT // 2)

    def update(self, speed=OBJ_SPEED):
        self.rect.y += speed