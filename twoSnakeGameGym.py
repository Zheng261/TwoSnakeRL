import gym
from gym import spaces
from random import randint
import numpy as np
import torch
from pygame.locals import *
import pygame
import time

## Size of game
GAME_WIDTH = 20
GAME_HEIGHT = 15
GAME_DEPTH = 1

## For rendering
BLOCK_SIZE = 44
YELLOW = (255, 255, 102)
GREEN = (0, 255, 0)

## Colors block for rendering stuff
def color_surface(surface, RGB):
    arr = pygame.surfarray.pixels3d(surface)
    arr[:,:,0] = RGB[0]
    arr[:,:,1] = RGB[1]
    arr[:,:,2] = RGB[2]

class Apple:
    def __init__(self,x,y):
        self.x = x 
        self.y = y 
    ## Draws apple in thing
    def draw(self, surface, image):
        surface.blit(image,(self.x * BLOCK_SIZE, self.y * BLOCK_SIZE)) 
 
class Player:
    def __init__(self, length, agent):
       self.x = []
       self.y = []
       self.length = length
       self.gameWidth = GAME_WIDTH
       self.gameHeight = GAME_HEIGHT
       self.agent = agent
       for i in range(0,1000):
           self.x.append(-100)
           self.y.append(-100)
       self.x[0] = randint(1,self.gameWidth-1)
       self.y[0] = randint(1,self.gameHeight-1)
       if length != 0:
        for i in range(length):
            self.length_update()


 
    def length_update(self):
      for i in range(self.length-1,0,-1):
            self.x[i] = self.x[i-1]
            self.y[i] = self.y[i-1]

    ### make RL friendly 
    def update(self, option):
        # update previous positions
        for i in range(self.length-1,0,-1):
            self.x[i] = self.x[i-1]
            self.y[i] = self.y[i-1]

        # update position of head of snake
        if option == 0:
            self.x[0] = self.x[0] + 1
        elif option == 1:
            self.x[0] = self.x[0] - 1
        elif option == 2:
            self.y[0] = self.y[0] - 1
        else:
            self.y[0] = self.y[0] + 1

    def draw(self, surface, image):
        for i in range(0,self.length):
            surface.blit(image,(self.x[i]*BLOCK_SIZE,self.y[i]*BLOCK_SIZE)) 


 
class Game:
    def isCollision(self,x1,y1,x2,y2):
        if x1 >= x2 and x1 <= x2:
            if y1 >= y2 and y1 <= y2:
                return True
        return False
 
class SnakeApp:
    def __init__(self):
        self._running = True
        self._display_surf = None
        self._image_surf = None
        self._apple_surf = None
        self.verbose = False

        self.game = Game()
        ### 2 agents
        agent_one = Player(3, agent=0)
        agent_two = Player(3, agent=1)
        self.players = [agent_one, agent_two]

        self.gameWidth = GAME_WIDTH
        self.gameHeight = GAME_HEIGHT
        self.gameDepth = GAME_DEPTH

        ## Initializes window width and windowheight
        self.windowWidth = BLOCK_SIZE * self.gameWidth
        self.windowHeight = BLOCK_SIZE * self.gameHeight

        self.apple = Apple(randint(1,self.gameWidth-1), randint(1,self.gameHeight-1))
        self.score = 0 

    ## Different grid based on agent since we do some feat. engineering
    def return_grid(self, agent=0):
        dist = np.linalg.norm(np.array([self.apple.x,self.apple.y]) - np.array([self.players[agent].x[0], self.players[agent].y[0]]))

        dist_to_other = np.linalg.norm(np.array([self.players[0].x[0], self.players[0].y[0]]) - np.array([self.players[1].x[0], self.players[1].y[0]]))
        xDiff_to_other = self.players[0].x[0] - self.players[1].x[0]
        yDiff_to_other = self.players[0].y[0] - self.players[1].y[0]

        xDiff = self.apple.x - self.players[agent].x[0]
        yDiff = self.apple.y - self.players[agent].y[0]

        leftDanger = 0
        rightDanger = 0
        upDanger = 0
        downDanger = 0
        if self.players[agent].x[0] == 1:
            leftDanger = 1
        elif self.players[agent].x[0] == self.gameWidth-1:
            rightDanger = 1
        if self.players[agent].y[0] == 1:
            downDanger = 1
        elif self.players[agent].y[0] == self.gameHeight-1:
            upDanger = 1

        ## Agents ramming into each other will kill both of them
        for i in range(0, self.players[0].length):
            if self.game.isCollision(self.players[agent].x[0]+1,self.players[agent].y[0],self.players[0].x[i], self.players[0].y[i]):
                rightDanger = 1
            if self.game.isCollision(self.players[agent].x[0]-1,self.players[agent].y[0],self.players[0].x[i], self.players[0].y[i]):
                leftDanger = 1
            if self.game.isCollision(self.players[agent].x[0],self.players[agent].y[0]+1,self.players[0].x[i], self.players[0].y[i]):
                upDanger = 1
            if self.game.isCollision(self.players[agent].x[0],self.players[agent].y[0]-1,self.players[0].x[i], self.players[0].y[i]):
                downDanger = 1

        for i in range(0, self.players[1].length):
            if self.game.isCollision(self.players[agent].x[0]+1,self.players[agent].y[0],self.players[1].x[i], self.players[1].y[i]):
                rightDanger = 1
            if self.game.isCollision(self.players[agent].x[0]-1,self.players[agent].y[0],self.players[1].x[i], self.players[1].y[i]):
                leftDanger = 1
            if self.game.isCollision(self.players[agent].x[0],self.players[agent].y[0]+1,self.players[1].x[i], self.players[1].y[i]):
                upDanger = 1
            if self.game.isCollision(self.players[agent].x[0],self.players[agent].y[0]-1,self.players[1].x[i], self.players[1].y[i]):
                downDanger = 1
        #print(agent, rightDanger, leftDanger, upDanger, downDanger)
        ### snake head, initialized badly
        return(np.array([dist, xDiff, yDiff, leftDanger, rightDanger, downDanger, upDanger, self.apple.x, self.apple.y, self.players[0].x[0], self.players[0].y[0], 
                self.players[0].x[1], self.players[0].y[1],self.players[0].x[2], self.players[0].y[2],
                self.players[1].x[0], self.players[1].y[0], self.players[1].x[1], self.players[1].y[1], self.players[1].x[2], self.players[1].y[2]]))
            
    def act(self, action, agent = 0):
        if self.verbose:
            print("agent ", agent , " did action ", action)
        self.players[agent].update(action)
        # does snake eat apple?
        if self.game.isCollision(self.apple.x,self.apple.y, self.players[agent].x[0], self.players[agent].y[0]):
            ### apple cannot spawn under snake
            viableAppleSpawns = []
            for i in range(1, self.gameWidth):
                for j in range(1, self.gameHeight):
                    viableAppleSpawns.append((i,j))

            for i in range(0,self.players[agent].length):
                if ((self.players[agent].x[i], self.players[agent].y[i]) in viableAppleSpawns):
                  viableAppleSpawns.remove((self.players[agent].x[i], self.players[agent].y[i]))

            appleSpawn = viableAppleSpawns[randint(0,len(viableAppleSpawns)-1)]
            self.apple.x = appleSpawn[0]
            self.apple.y = appleSpawn[1]
            self.players[agent].length = self.players[agent].length + 1
            self.players[agent].length_update()

            self.score = self.score + 1000

            return (self.return_grid(agent), 1000, False, {"episode": {"r": 1000}})

        # does snake collide with itself?
        for i in range(0,self.players[0].length):
            if (i != 0 or agent != 0):
                if self.game.isCollision(self.players[agent].x[0],self.players[agent].y[0],self.players[0].x[i], self.players[0].y[i]):
                    if agent == 0:
                        if self.verbose:
                            print("dead, self collide, we are ", agent)
                    else:
                        if self.verbose:
                            print("dead, collide with other agent, we are ", agent)
                    return (self.return_grid(agent), -1, True, {"episode": {"r": -1}})

        # does snake collide with other one?
        for i in range(0,self.players[1].length):
            if (i != 0 or agent != 1):
                if self.game.isCollision(self.players[agent].x[0],self.players[agent].y[0],self.players[1].x[i], self.players[1].y[i]):
                    if agent == 1:
                        if self.verbose:
                            print("dead, self collide, we are ", agent)
                    else:
                        if self.verbose:
                            print("dead, collide with other agent, we are ", agent)
                    return (self.return_grid(agent), -1, True, {"episode": {"r": -1}})

        # left/right boundary
        if (self.players[agent].x[0] <= 0 or self.players[agent].x[0] >= self.gameWidth):
            if self.verbose:
                print("dead, LR bounds")
       
            return (self.return_grid(agent), -1, True,  {"episode": {"r": -1}})

        # top/bottom boundary
        if (self.players[agent].y[0] <= 0 or self.players[agent].y[0] >= self.gameHeight):
            if self.verbose:
                print("dead, UB bounds")
           
            return (self.return_grid(agent), -1, True,  {"episode": {"r": -1}})
            
        # Score goes down by 1 every turn to discourage inefficiency
        self.score = self.score - 1
        return (self.return_grid(agent), - 1, False, {"episode": {"r": -1}})


            ### Purely for graphics
    def on_init(self):
        print("Initialized")
        pygame.init()
        self._running = True
        self._display_surf = pygame.display.set_mode((self.windowWidth,self.windowHeight), pygame.HWSURFACE)
        self.score_font = pygame.font.SysFont("comicsansms", 30)
        pygame.display.set_caption('Snake RL Test')
        self._image_surf = pygame.image.load("block.png").convert()
        self._image_surf = pygame.transform.scale(self._image_surf,(BLOCK_SIZE,BLOCK_SIZE))
        self._apple_surf = pygame.image.load("apple.png").convert()
        self._apple_surf = pygame.transform.scale(self._apple_surf,(BLOCK_SIZE,BLOCK_SIZE))
        #self.show_score(self.score)
 
    def on_event(self, event):
        if event.type == QUIT:
            self._running = False
     
    def show_score(self, score):
        value = self.score_font.render("Your Score: " + str(score), True, YELLOW)
        self._display_surf.blit(value, [0, 0])

    def on_render(self):
        print("Rendering!")
        self._display_surf.fill((2,2,2))
        #self.show_score(self.score)
        self.players[0].draw(self._display_surf, self._image_surf)
        self.players[1].draw(self._display_surf, self._image_surf)
        self.apple.draw(self._display_surf, self._apple_surf)
        pygame.display.update()

    def on_cleanup(self):
        pygame.quit()

    ################

class TwoSnakeGameGym(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, initBoard=False):
    super(TwoSnakeGameGym, self).__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:
    self.action_space = spaces.Discrete(4)
    self.initBoard = initBoard
    # Example for using image as input:
    self.observation_space = spaces.Box(
      low=0, high=21, shape=([21]), dtype=np.float16)
    self.SnakeApp = SnakeApp()
    if self.initBoard:
        self.SnakeApp.on_init()

  def step(self, action, agent = 0):
    ## Needed for rendering if we want the GUI
    if self.initBoard:
        pygame.event.pump()
    return self.SnakeApp.act(action, agent)
    # Execute one time step within the environment
    ...
  def reset(self):
    self.SnakeApp.on_cleanup()
    self.SnakeApp = SnakeApp()
    if self.initBoard:
        self.SnakeApp.on_init()
    ## Returns init. state for both agent 0 and 1
    return [self.SnakeApp.return_grid(0), self.SnakeApp.return_grid(1)]
    # Reset the state of the environment to an initial state
    ...

  def render(self, mode='human', close=False):
    self.SnakeApp.on_render()
    # Render the environment to the screen
    ...
  def cleanup(self):
    self.SnakeApp.on_cleanup()
    

