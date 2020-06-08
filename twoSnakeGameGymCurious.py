import gym
from gym import spaces
from random import randint
import numpy as np
import torch
from pygame.locals import *
import pygame
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)
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
 
class Network(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        input_size = input_size
        hidden_sizes = [30,20]
        output_size = 4
        
        # Inputs to hidden layer linear transformation
        self.model = torch.nn.Sequential(torch.nn.Linear(input_size, hidden_sizes[0]),
                     torch.nn.ReLU(),
                     torch.nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                     torch.nn.ReLU(),
                     torch.nn.Linear(hidden_sizes[1], output_size),
                     torch.nn.Softmax(dim=0))
        
    def forward(self, x):
        # Pass the input tensor through each of our operations
        return self.model(x)


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

        self.intrinsicMotivation = True

        ### This never gets updated if intrinsic motivation is false
        self.intrinsicReward = 0

        if self.intrinsicMotivation:
            ## How much our curiosity is weighted
            self.curiosityWeight = 50
            
            self.CeLoss = torch.nn.CrossEntropyLoss()
            ## If our action causes other agent to do something, or does something that takes the other agent into account
            ## Then reward that action
            ## Module takes in which agent it's predicting for: ourselves, or another agent
            ## Module takes in whether it's predicting the other agent's action given the state, or trying to predict our action given the state

            ## We want to encourage our own actions that either 1) influence the other agent, or 2) was influenced by the other agent

            ## We run this guess right after we act and keep track of our last guesses 
            self.avgLoss = 0
            self.iters = 0
            self.ActionGuesserModuleNoState = Network(8).to(device)
            ### We take our old output and pass it through a new NN which augments the probs with new info
            self.ActionGuesserModuleWithState = Network(15).to(device)

            self.noStateOptimizer = torch.optim.Adam(self.ActionGuesserModuleNoState.parameters(), lr=0.001)
            self.StateOptimizer = torch.optim.Adam(self.ActionGuesserModuleWithState.parameters(), lr=0.001)

            self.firstAgentLastGuess = -1
            self.firstAgentLastGuessNoState = -1

            self.secondAgentLastGuessNoState = -1
            self.secondAgentLastGuess = -1

            self.withStateValues = []
            self.noStateValues = []
            self.withStateValuesSelf = []
            self.noStateValuesSelf= []

            ## Guesses what an agent does, given ALL available information except our own state/action.
            self.firstAgentLastAction = -1
            self.secondAgentLastAction = -1

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

        if self.intrinsicMotivation:

            ### Add reward based on how much our last action ended up being used by the other agent
            #if self.firstAgentLastAction != -1 and self.secondAgentLastAction != -1:

                #print("adding in int. rew for agent:", agent)
            if agent == 0:
                lastGuess = self.firstAgentLastGuess
                lastGuessNoState =  self.firstAgentLastGuessNoState
                lastAgentAction = self.firstAgentLastAction
                lastOtherAgentAction = self.secondAgentLastAction
            if agent == 1:
                lastGuess = self.secondAgentLastGuess
                lastGuessNoState =  self.secondAgentLastGuessNoState
                lastAgentAction = self.secondAgentLastAction
                lastOtherAgentAction = self.firstAgentLastAction

            ### Training
            if False:
                stateLossOne = self.CeLoss(lastGuess.unsqueeze(0), torch.tensor([lastOtherAgentAction]))
                noStateLossOne = self.CeLoss(lastGuessNoState.unsqueeze(0), torch.tensor([lastOtherAgentAction]))
              
                ### We give reward if, last round, we see that our action influenced the other agent in some way.
                ### This is true if we have a higher pred. acc.
                ### for their action if we include our own action/state than if we did not.
                self.intrinsicReward = self.curiosityWeight * max(lastGuess[lastOtherAgentAction].item() - lastGuessNoState[lastOtherAgentAction].item(), 0)
                
            ### Next, we predict what our action will be with/without the other agent. We reward ourselves for taking an action
            ### That involves the other agent (pred. accuracy higher given their action and position)
            isPredictingSelfAction = 1
           
            otherAgent = -1*agent + 1

            ###  What will we do without the other agent -- our action pred. only sees ourselves?
            self.noStateValuesSelf = [self.apple.x, self.apple.y, self.players[agent].x[0], self.players[agent].y[0], self.gameWidth, self.gameHeight, agent, isPredictingSelfAction]
            guessNoState = self.ActionGuesserModuleNoState(torch.tensor(self.noStateValuesSelf).double())

            self.withStateValuesSelf = [self.apple.x, self.apple.y, self.players[agent].x[0], self.players[agent].y[0], self.gameWidth, self.gameHeight, agent, isPredictingSelfAction, lastOtherAgentAction, self.players[otherAgent].x[0], self.players[otherAgent].y[0]]
            ### How does the presence/action of the other agent change our behavior?
            guessNoGradWithState = torch.cat((guessNoState.detach().double(), torch.tensor(self.withStateValuesSelf).double()), 0)
            guessWithState = self.ActionGuesserModuleWithState(guessNoGradWithState)

            stateLossTwo = self.CeLoss(guessWithState.unsqueeze(0), torch.tensor([action]))
            noStateLossTwo = self.CeLoss(guessNoState.unsqueeze(0), torch.tensor([action]))

            ### Training
            self.noStateOptimizer.zero_grad()
            #noStateLossOne.mean().backward()
            noStateLossTwo.mean().backward()
            self.noStateOptimizer.step()

            self.StateOptimizer.zero_grad()
            #stateLossOne.mean().backward()
            stateLossTwo.mean().backward()
            self.StateOptimizer.step()
            self.avgLoss += stateLossTwo.mean().item()
            self.iters += 1
            if (self.iters % 100 == 1):
                print(self.avgLoss/self.iters)

            ### We give reward if, last round, we see that our action influenced the other agent in some way. This is true if we have a higher pred. acc.
            ### for our action if we include our own action/state than if we did not.
            self.intrinsicReward = self.intrinsicReward + self.curiosityWeight * max(guessWithState[action].item() - guessNoState[action].item(), 0)
            #print(self.intrinsicReward)
        #print("agent ", agent , " did action ", action)

        if self.verbose:
            print("agent ", agent , " did action ", action)
        self.players[agent].update(action)

        ### Predicts what other agent will do. Will reward next time if it turns out that our action influenced them
        ### Action influences other agent IF we have a higher pred. accuracy for other agent, given our action/state, than not
        if agent == 0:
            self.firstAgentLastAction = action
        if agent == 1:
            self.secondAgentLastAction = action

        if False:
            if self.intrinsicMotivation:
                print("triggered stuff for agent :", agent)
                isPredictingSelfAction = 0
                if agent == 0:
                    ### What would snake 1 do without the current agent (snake 0)?
                    self.firstAgentLastAction = action
                    self.noStateValues = [self.apple.x, self.apple.y, self.players[1].x[0], self.players[1].y[0], self.gameWidth, self.gameHeight, agent, isPredictingSelfAction]
                    self.firstAgentLastGuessNoState = self.ActionGuesserModuleNoState(torch.tensor(self.noStateValues).double())
                    ### Passes our guess so far + context with action into the next module -- what would snake 1 do given the current agent (snake 0)?
                    self.withStateValues = [self.apple.x, self.apple.y, self.players[1].x[0], self.players[1].y[0], self.gameWidth, self.gameHeight, agent, isPredictingSelfAction, action, self.players[0].x[0], self.players[0].y[0]]

                    lastGuessNoGradWithState = torch.cat((self.firstAgentLastGuessNoState.detach().double(), torch.tensor(self.withStateValues).double()), 0)
                    self.firstAgentLastGuess= self.ActionGuesserModuleWithState(lastGuessNoGradWithState)

                if agent == 1:

                    self.secondAgentLastAction = action
                    ### What would snake 0 do without the current agent (snake 1)?
                    self.noStateValues = [self.apple.x, self.apple.y, self.players[0].x[0], self.players[0].y[0], self.gameWidth, self.gameHeight, agent, isPredictingSelfAction]
                  

                    self.secondAgentLastGuessNoState = self.ActionGuesserModuleNoState(torch.tensor(self.noStateValues).double())


                    self.withStateValues = [self.apple.x, self.apple.y, self.players[0].x[0], self.players[0].y[0], self.gameWidth, self.gameHeight, agent, isPredictingSelfAction, action, self.players[1].x[0], self.players[1].y[0]]

                     ### Passes our guess so far + context with action into the next module -- What would snake 0 do now given the current agent (snake 1)?

                    lastGuessNoGradWithState = torch.cat((self.secondAgentLastGuessNoState.detach().double(), torch.tensor(self.withStateValues).double()), 0)
                    self.secondAgentLastGuess = self.ActionGuesserModuleWithState(lastGuessNoGradWithState)
                   

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

            return (self.return_grid(agent), 1000+self.intrinsicReward, False, {"episode": {"r": 1000}})

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
                    return (self.return_grid(agent), -1+self.intrinsicReward, True, {"episode": {"r": -1}})

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
                    return (self.return_grid(agent), -1+self.intrinsicReward, True, {"episode": {"r": -1}})

        # left/right boundary
        if (self.players[agent].x[0] <= 0 or self.players[agent].x[0] >= self.gameWidth):
            if self.verbose:
                print("dead, LR bounds")
       
            return (self.return_grid(agent), -1+self.intrinsicReward, True,  {"episode": {"r": -1}})

        # top/bottom boundary
        if (self.players[agent].y[0] <= 0 or self.players[agent].y[0] >= self.gameHeight):
            if self.verbose:
                print("dead, UB bounds")
           
            return (self.return_grid(agent), -1+self.intrinsicReward, True,  {"episode": {"r": -1}})
            
        # Score goes down by 1 every turn to discourage inefficiency
        self.score = self.score - 1
        return (self.return_grid(agent), -10+self.intrinsicReward, False, {"episode": {"r": -1}})


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

    def reset(self):
        ## Resets game completely, besides the NN
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

        ### Intrinsic motivation
        if self.intrinsicMotivation:
            ## How much our curiosity is weighted
            self.noStateOptimizer = torch.optim.Adam(self.ActionGuesserModuleNoState.parameters(), lr=0.001)
            self.StateOptimizer = torch.optim.Adam(self.ActionGuesserModuleWithState.parameters(), lr=0.001)

            self.CeLoss = torch.nn.CrossEntropyLoss()
            ## If our action causes other agent to do something 
            ## Then reward that action
            ## Guesses what an agent does, given ALL available information 
            self.firstAgentLastGuess = -1
            self.firstAgentLastGuessNoState = -1

            self.secondAgentLastGuessNoState = -1
            self.secondAgentLastGuess = -1

            ## Guesses what an agent does, given ALL available information except our own state/action.
            self.firstAgentLastAction = -1
            self.secondAgentLastAction = -1

            self.intrinsicReward = 0




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
    self.SnakeApp.reset()
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
    

