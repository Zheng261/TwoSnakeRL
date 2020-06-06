from pygame.locals import *
from random import randint
import pygame
import time
 

BLOCK_SIZE = 44
YELLOW = (255, 255, 102)
GREEN = (0, 255, 0)

def color_surface(surface, RGB):
    arr = pygame.surfarray.pixels3d(surface)
    arr[:,:,0] = RGB[0]
    arr[:,:,1] = RGB[1]
    arr[:,:,2] = RGB[2]

class Apple:
    def __init__(self,x,y):
        self.x = 5
        self.y = 5
    def draw(self, surface, image):
        surface.blit(image,(self.x * BLOCK_SIZE, self.y * BLOCK_SIZE)) 
 
 
class Player:
    x = [0]
    y = [0]
    direction = 0 
    updateCountMax = 2
    updateCount = 0
 
    def __init__(self, length):
       self.length = length
       self.gameWidth = 20
       self.gameHeight = 15
       for i in range(0,1000):
           self.x.append(-100)
           self.y.append(-100)
 
       # initial positions, no collision.
       self.x[0] = randint(1,self.gameWidth-1)
       self.y[0] = randint(1,self.gameHeight-1)

    def update(self):
        self.updateCount = self.updateCount + 1
        if self.updateCount > self.updateCountMax:
 
            # update previous positions
            for i in range(self.length-1,0,-1):
                self.x[i] = self.x[i-1]
                self.y[i] = self.y[i-1]
 
            # update position of head of snake
            if self.direction == 0:
                self.x[0] = self.x[0] + 1
            if self.direction == 1:
                self.x[0] = self.x[0] - 1
            if self.direction == 2:
                self.y[0] = self.y[0] - 1
            if self.direction == 3:
                self.y[0] = self.y[0] + 1
 
            self.updateCount = 0
 
 
    def moveRight(self):
        self.direction = 0
 
    def moveLeft(self):
        self.direction = 1
 
    def moveUp(self):
        self.direction = 2
 
    def moveDown(self):
        self.direction = 3 
 
    def draw(self, surface, image):
        for i in range(0,self.length):
            surface.blit(image,(self.x[i]*BLOCK_SIZE,self.y[i]*BLOCK_SIZE)) 
 
class Game:
    def isCollision(self,x1,y1,x2,y2):
        if x1 >= x2 and x1 <= x2:
            if y1 >= y2 and y1 <= y2:
                return True
        return False
 
class App:
    player = 0
    apple = 0
    

    def __init__(self):
        self._running = True
        self._display_surf = None
        self._image_surf = None
        self._apple_surf = None
        self.game = Game()
        self.player = Player(1) 
        self.apple = Apple(5,5)
        self.gameWidth = 20
        self.gameHeight = 15
        self.windowWidth = BLOCK_SIZE * self.gameWidth
        self.windowHeight = BLOCK_SIZE * self.gameHeight
        self.score = 0

    def on_init(self):
        pygame.init()
        self._display_surf = pygame.display.set_mode((self.windowWidth,self.windowHeight), pygame.HWSURFACE)
        self.score_font = pygame.font.SysFont("comicsansms", 30)
        pygame.display.set_caption('Pygame pythonspot.com example')
        self._running = True
        self._image_surf = pygame.image.load("block.png").convert()
        self._image_surf = pygame.transform.scale(self._image_surf,(BLOCK_SIZE,BLOCK_SIZE))
        self._apple_surf = pygame.image.load("apple.png").convert()
        self._apple_surf = pygame.transform.scale(self._apple_surf,(BLOCK_SIZE,BLOCK_SIZE))
        self.show_score(self.score)
        
 
    def on_event(self, event):
        if event.type == QUIT:
            self._running = False
     
    def show_score(self, score):
        value = self.score_font.render("Your Score: " + str(score), True, YELLOW)
        self._display_surf.blit(value, [0, 0])

    def on_loop(self):
        self.player.update()
        self.show_score(self.score)
        # does snake eat apple?
        
        if self.game.isCollision(self.apple.x,self.apple.y, self.player.x[0], self.player.y[0]):
            ### apple cannot spawn under snake
            viableAppleSpawns = []
            for i in range(1, self.gameWidth):
                for j in range(1, self.gameHeight):
                    viableAppleSpawns.append((i,j))

            for i in range(0,self.player.length):
                if ((self.player.x[i], self.player.y[i]) in viableAppleSpawns):
                    viableAppleSpawns.remove((self.player.x[i], self.player.y[i]))

            appleSpawn = viableAppleSpawns[randint(0,len(viableAppleSpawns)-1)]
            self.apple.x = appleSpawn[0]
            self.apple.y = appleSpawn[1]

            self.player.length = self.player.length + 1
            self.score = self.score + 1
                
        # does snake collide with itself?
        for i in range(2,self.player.length):
            if self.game.isCollision(self.player.x[0],self.player.y[0],self.player.x[i], self.player.y[i]):
                print("You lose! Collision: ")
                print("x[0] (" + str(self.player.x[0]) + "," + str(self.player.y[0]) + ")")
                print("x[" + str(i) + "] (" + str(self.player.x[i]) + "," + str(self.player.y[i]) + ")")
                exit(0)

        if (self.player.x[0] < 0 or self.player.x[0] > self.gameWidth):
            print("You lose! Left/Right Boundaries")
            print("x[0] (" + str(self.player.x[0]) + "," + str(self.player.y[0]) + ")")
            exit(0)

        if (self.player.y[0] < 0 or self.player.y[0] > self.gameHeight):
            print("You lose! Top/Bottom Boundaries")
            print("x[0] (" + str(self.player.x[0]) + "," + str(self.player.y[0]) + ")")
            exit(0)
 
        pass
 
    def on_render(self):
        self._display_surf.fill((0,0,0))
        self.show_score(self.score)
        self.player.draw(self._display_surf, self._image_surf)
        self.apple.draw(self._display_surf, self._apple_surf)
        pygame.display.update()
 
    def on_cleanup(self):
        pygame.quit()
 
    def on_execute(self):
        if self.on_init() == False:
            self._running = False
        
        while( self._running ):
            pygame.event.pump()
            keys = pygame.key.get_pressed() 
 
            if (keys[K_RIGHT]):
                self.player.moveRight()
 
            if (keys[K_LEFT]):
                self.player.moveLeft()
 
            if (keys[K_UP]):
                self.player.moveUp()
 
            if (keys[K_DOWN]):
                self.player.moveDown()
 
            if (keys[K_ESCAPE]):
                self._running = False
 
            self.on_loop()
            self.on_render()
 
            time.sleep (50.0 / 1000.0);
        self.on_cleanup()
 
if __name__ == "__main__" :
    theApp = App()
    theApp.on_execute()