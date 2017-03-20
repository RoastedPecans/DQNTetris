# Program Name: Tetris.py
# Program Purpose: Tetris built in Python for use with a DQN that will learn to play
# Date Started: 1/3/17
# Last Modified: 3/18/17
# Programmer: Connor

import tensorflow as tf
from collections import deque  # Deque used for replay memory
import pygame
import numpy
from pygame.locals import *
from pygame import event
import random
import time
from PIL import Image
from copy import deepcopy
import datetime

pygame.init()
pygame.mixer.init()
linesClearedSound = pygame.mixer.Sound("dingSoundEffect.wav")

# Global Variables for game
score = 0  # Total score
thetaScore = 1000  # Keep track of changes in score (used to reward agent)
thetaScore2 = 1000  #  Use to make one reward per piece
level = 1  # Start on level 1
frameName = 0  # Used for naming screenshots
terminal = False  # Flag for GameOver
linesCleared = 0
MAX_REWARD = 0
reward = 0
newBoard = []
gamesPlayed = 0
FONT_PATH = "C:\Windows\Fonts\Arial.ttf"  # Use C:\Windows\Fonts\Arial.ttf for Windows, /System/Library/Fonts/Helvetica.dfont for Mac.
FONT = pygame.font.Font(FONT_PATH, 12)

startTime = datetime.datetime.now()

class Piece:
    O = (((0, 0, 0, 0, 0),  (0, 0, 0, 0, 0),  (0, 0, 1, 1, 0),  (0, 0, 1, 1, 0),  (0, 0, 0, 0, 0)), ) * 4  # Square Tetromino

    I = (((0, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 1, 1, 1, 1), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)),   # Straight Tetromino
         ((0, 0, 0, 0, 0), (0, 0, 1, 0, 0), (0, 0, 1, 0, 0), (0, 0, 1, 0, 0), (0, 0, 1, 0, 0)),   # [][][][]
         ((0, 0, 0, 0, 0), (0, 0, 0, 0, 0), (1, 1, 1, 1, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)),
         ((0, 0, 1, 0, 0), (0, 0, 1, 0, 0), (0, 0, 1, 0, 0), (0, 0, 1, 0, 0), (0, 0, 0, 0, 0)))

    L = (((0, 0, 0, 0, 0), (0, 0, 1, 0, 0), (0, 0, 1, 0, 0), (0, 0, 1, 1, 0), (0, 0, 0, 0, 0)),   # L-shaped Tetromino [][][]
         ((0, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 1, 1, 1, 0), (0, 1, 0, 0, 0), (0, 0, 0, 0, 0)),   #                   []
         ((0, 0, 0, 0, 0), (0, 1, 1, 0, 0), (0, 0, 1, 0, 0), (0, 0, 1, 0, 0), (0, 0, 0, 0, 0)),
         ((0, 0, 0, 0, 0), (0, 0, 0, 1, 0), (0, 1, 1, 1, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)))

    J = (((0, 0, 0, 0, 0), (0, 0, 1, 0, 0), (0, 0, 1, 0, 0), (0, 1, 1, 0, 0), (0, 0, 0, 0, 0)),   # Opposite of L Tetromino []
         ((0, 0, 0, 0, 0), (0, 1, 0, 0, 0), (0, 1, 1, 1, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)),   #                    [][][]
         ((0, 0, 0, 0, 0), (0, 0, 1, 1, 0), (0, 0, 1, 0, 0), (0, 0, 1, 0, 0), (0, 0, 0, 0, 0)),
         ((0, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 1, 1, 1, 0), (0, 0, 0, 1, 0), (0, 0, 0, 0, 0)))

    Z = (((0, 0, 0, 0, 0), (0, 0, 0, 1, 0), (0, 0, 1, 1, 0), (0, 0, 1, 0, 0), (0, 0, 0, 0, 0)),   # Z-shaped Tetromino [][]
         ((0, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 1, 1, 0, 0), (0, 0, 1, 1, 0), (0, 0, 0, 0, 0)),   #                     [][]
         ((0, 0, 0, 0, 0), (0, 0, 1, 0, 0), (0, 1, 1, 0, 0), (0, 1, 0, 0, 0), (0, 0, 0, 0, 0)),
         ((0, 0, 0, 0, 0), (0, 1, 1, 0, 0), (0, 0, 1, 1, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)))

    S = (((0, 0, 0, 0, 0), (0, 0, 1, 0, 0), (0, 0, 1, 1, 0), (0, 0, 0, 1, 0), (0, 0, 0, 0, 0)),   # S-shaped Tetromino [][]
         ((0, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 1, 1, 0), (0, 1, 1, 0, 0), (0, 0, 0, 0, 0)),   #                 [][]
         ((0, 0, 0, 0, 0), (0, 1, 0, 0, 0), (0, 1, 1, 0, 0), (0, 0, 1, 0, 0), (0, 0, 0, 0, 0)),
         ((0, 0, 0, 0, 0), (0, 0, 1, 1, 0), (0, 1, 1, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)))

    T = (((0, 0, 0, 0, 0), (0, 0, 1, 0, 0), (0, 0, 1, 1, 0), (0, 0, 1, 0, 0), (0, 0, 0, 0, 0)),   # T-shaped Tetromino [][][]
         ((0, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 1, 1, 1, 0), (0, 0, 1, 0, 0), (0, 0, 0, 0, 0)),   #                     []
         ((0, 0, 0, 0, 0), (0, 0, 1, 0, 0), (0, 1, 1, 0, 0), (0, 0, 1, 0, 0), (0, 0, 0, 0, 0)),
         ((0, 0, 0, 0, 0), (0, 0, 1, 0, 0), (0, 1, 1, 1, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)))

    PIECES = {'O': O, 'I': I, 'L': L, 'J': J, 'Z': Z, 'S': S, 'T': T}  # Create a dictionary to hold Pieces

    def __init__(self, piece_name=None):
        if piece_name:
            self.piece_name = piece_name
        else:
            self.piece_name = random.choice(list(Piece.PIECES.keys()))  # If no "first piece" then randomly select from a list of the keys in PIECES dictionary
        self.rotation = 0
        self.array2d = Piece.PIECES[self.piece_name][self.rotation]  # In array2d save the chosen Tetromino with the set rotation

    def __iter__(self):
        for row in self.array2d:
            yield row   # Yield is the same as return but it returns a generator instead of an iterative

    def rotate(self, clockwise=True):
        self.rotation = (self.rotation + 1) % 4 if clockwise else \
            (self.rotation - 1) % 4   # Add 1 for clockwise, subtract 1 for counterclockwise
        self.array2d = Piece.PIECES[self.piece_name][self.rotation]


class Board:

    # Get board rows for earlier frame, compare to new frame using bitwise (same (1 +1) = 0), new will be 1 + 0 = 1
    COLLIDE_ERROR = {'no_error': 0,  'right_wall': 1, 'left_wall': 2, 'bottom': 3, 'overlap': 4}  # Dictionary for storing what kind of collision occurred

    def generatePieceWithScore(self):
        # Update to current board for next loop, but feed old one for comparision this loop

        self.piece = Piece()   # Set first piece to random piece per the Piece Class init function
        self.piece_x, self.piece_y = 3, 0   # Set to center

    def calcScore(self):
        global thetaScore, thetaScore2, newBoard, score, FONT, MAX_REWARD, reward

        # Create blank board to hold changes between frames
        thetaBoard = []
        for i in range(self.height):
            thetaBoard.append([0] * self.width)

        # Compare two boards to calculate reward
        for i in range(self.height):
            for j in range(self.width):
                if newBoard[i][j] is not self.board[i][j] and i != 0:
                    thetaBoard[i][j] = (1 / i) * 50  # Where there has been a change, set index to value

        # Create copy by value, not reference
        newBoard = deepcopy(self.board)
        thetaScore -= round((numpy.sum(thetaBoard)))  # Calculate total score for that piece

        #print('REWARD: ' + str(thetaScore))

        # Update GUI to show total reward so far
        white = (255, 255, 255)
        black = (0, 0, 0)

        if thetaScore > MAX_REWARD:
            MAX_REWARD = thetaScore
            label = FONT.render("Max Reward: " + str(MAX_REWARD), 1, white, black)
            self.surface.blit(label, (0, 530))

        score += round(thetaScore)  # Keep track of total reward earned
        reward = thetaScore
        
        label = FONT.render("Score: " + str(score),  1, white, black)
        self.surface.blit(label, (0,  510))
        thetaScore2 = thetaScore

    def __init__(self, surface):
        global newBoard
        self.surface = surface
        self.width = 10   # Width of game board
        self.height = 22   # Height of game board
        self.block_size = 25   # "Block Size" of grid squares in pixels
        self.board = []
        for x in range(self.height):   # For all "points" on board, set to 0
            self.board.append([0] * self.width)
            newBoard.append([0] * self.width)
        self.generatePieceWithScore()   # Generate first piece

    def absorb_piece(self):
        for y, row in enumerate(self.piece):  # For all rows
            for x, block in enumerate(row):
                if block:
                    self.board[y+self.piece_y][x+self.piece_x] = block
        self.generatePieceWithScore()

    def _block_collide_with_board(self, x, y):
        if x < 0:
            return Board.COLLIDE_ERROR['left_wall']
        elif x >= self.width:
            return Board.COLLIDE_ERROR['right_wall']
        elif y >= self.height:
            return Board.COLLIDE_ERROR['bottom']
        elif self.board[y][x]:
            return Board.COLLIDE_ERROR['overlap']
        return Board.COLLIDE_ERROR['no_error']

    def collide_with_board(self, dx, dy):
        for y, row in enumerate(self.piece):
            for x, block in enumerate(row):
                if block:
                    collide = self._block_collide_with_board(x=x+dx, y=y+dy)
                    if collide:
                        return collide
        return Board.COLLIDE_ERROR['no_error']

    def _can_move_piece(self, dx, dy):
        dx_ = self.piece_x + dx
        dy_ = self.piece_y + dy
        if self.collide_with_board(dx=dx_, dy=dy_):
            return False
        return True

    def _try_rotate_piece(self, clockwise=True):
        self.piece.rotate(clockwise)
        collide = self.collide_with_board(dx=self.piece_x, dy=self.piece_y)
        if not collide:
            pass
        elif collide == Board.COLLIDE_ERROR['left_wall']:
            if self._can_move_piece(dx=1, dy=0):
                self.move_piece(dx=1, dy=0)
            elif self._can_move_piece(dx=2, dy=0):
                self.move_piece(dx=2, dy=0)
            else:
                self.piece.rotate(not clockwise)
        elif collide == Board.COLLIDE_ERROR['right_wall']:
            if self._can_move_piece(dx=-1, dy=0):
                self.move_piece(dx=-1, dy=0)
            elif self._can_move_piece(dx=-2, dy=0):
                self.move_piece(dx=-2, dy=0)
            else:
                self.piece.rotate(not clockwise)
        else:
            self.piece.rotate(not clockwise)

    def move_piece(self, dx, dy):
        if self._can_move_piece(dx, dy):
            self.piece_x += dx
            self.piece_y += dy

    def _can_drop_piece(self):
        return self._can_move_piece(dx=0,  dy=1)

    def rotate_piece(self, clockwise=True):
        self._try_rotate_piece(clockwise)

    def pos_to_pixel(self, x ,y):
        return self.block_size* x, self.block_size*(y-2)

    def _delete_line(self, y):
        for y in reversed(range(1, y+1)):  # Start by clearing top row first
            self.board[y] = list(self.board[y-1])

    def delete_lines(self):
        global thetaScore, linesCleared, FONT
        remove = [y for y, row in enumerate(self.board) if all(row)]
        if len(remove) > 0:
            thetaScore += len(remove) * 10000  # 10000 reward per line cleared
            linesClearedSound.play()  # Play sound!
            # Update GUI
            linesCleared += len(remove)
            label = FONT.render("Lines cleared: " + str(linesCleared), 1, (255, 255, 255), (0, 0, 0))
            self.surface.blit(label, (125, 510))
        for y in remove:
            self._delete_line(y)

    """
    def score(self, scoreToAdd):
        global score
        global level
        global thetaScore
        print("Score to Add: " + str(scoreToAdd))
        if scoreToAdd == 1:
            scoreToAdd = 40 * level
        elif scoreToAdd == 2:
            scoreToAdd = 100 * level
        elif scoreToAdd == 3:
            scoreToAdd = 300 * level
        elif scoreToAdd == 4:
            scoreToAdd = 1200 * level
        else:
            print("ELSE")  # Should never be called unless agent somehow performs god-like actions and clears 5 lines at once
        thetaScore = scoreToAdd  # make a copy of the score being added to determine change in score (send as reward to agent)
        print("ThetaScore: " + str(thetaScore))
        score += scoreToAdd
        white = (255, 255, 255)
        black = (0, 0, 0)
        label = self.font.render("Score: " + str(score),  1, white, black)
        self.surface.blit(label, (0,  530))
        print(score)
        # self.levelUp()  # Every time score is updated see if we level up. Currently deprecated.
    # Currently deprecated as it may cause agent to learn less by emphasizing late-game actions.
    def levelUp(self):
        global level
        global score
        scoreThreshold = 1000 * level
        if score >= scoreThreshold and level < 10:  # Increment score if pass the scorethreshold and under level 10
            level += 1
            print("Level Up!")
            white = (255, 255, 255)
            black = (0, 0, 0)
            levelLabel = self.font.render("Level: " + str(level), 1, white, black)
            self.surface.blit(levelLabel, (0, 500))
    """

    def game_over(self):
        return sum(self.board[0]) > 0 or sum(self.board[1]) > 0

    def draw_blocks(self, array2d, color=(0, 0, 255),  dx=0,  dy=0):
        for y, row in enumerate(array2d):
            y += dy
            if y >= 2 and y < self.height:
                for x, block in enumerate(row):
                    if block:
                        #If there's a grid block to be drawn, draw it
                        self.delete_lines()  # See if we cleared any lines
                        x += dx
                        x_pix, y_pix = self.pos_to_pixel(x, y)
                         # draw block
                        pygame.draw.rect(self.surface, color,
                                         (  x_pix, y_pix,
                                            self.block_size,
                                            self.block_size))
                         # draw border
                        pygame.draw.rect(self.surface, (0,  0,  0),
                                         (  x_pix, y_pix,
                                            self.block_size,
                                            self.block_size),  1)

    def draw(self):

        self.draw_blocks(self.piece, dx=self.piece_x, dy=self.piece_y)  # Draws the piece
        self.draw_blocks(self.board)  # Redraws the whole board

    def drop_piece(self):
        if self.game_over():
            self.resetBoard()
        if self._can_drop_piece():
            self.move_piece(dx=0, dy=1)
        else:
            self.absorb_piece()
            self.delete_lines()

    def resetBoard(self):
        print("Resetting game!")
        global level, terminal, newBoard, FONT, gamesPlayed, thetaScore, thetaScore2
        terminal = True

        thetaScore = 1000
        thetaScore2 = 1000
        
        gamesPlayed += 1
        label = FONT.render('Games Played: ' + str(gamesPlayed), 1, (255, 255, 255), (0, 0, 0))
        self.surface.blit(label, (125, 530))
        
        # Reset boards
        self.board = []
        newBoard = []
        for x in range(self.height):
            self.board.append([0] * self.width)
            newBoard.append([0] * self.width)


class Tetris:
    DROP_EVENT = USEREVENT + 1
    frameNumber = 0  # Will be used to count number of frames.
    frameStack = []  # Will be used to hold sequences of 4 frames

    def __init__(self):
        self.surface = pygame.display.set_mode((250, 550))  # Set dimensions of game window. Creates a Surface
        self.clock = pygame.time.Clock()  # Create game clock
        self.board = Board(self.surface)  # Create board for Tetris pieces
        pygame.display.update()

    def handle_input(self, agentInput):
        pygame.event.pump()  # Needs to be called every frame so pygame can interact with OS
        global thetaScore, thetaScore2, frameName, score, FONT, terminal, MAX_REWARD, reward
        
        # Before running action set terminal (game over) to False
        if terminal:
            terminal = False

        # Do nothing[0], rotate right[1], rotate left[2], move left[3], move right [4], drop piece to bottom[5].
        if agentInput[1] == 1:
            self.board.move_piece(-1, 0)
        elif agentInput[2] == 1:
            self.board.rotate_piece(clockwise=False)
        elif agentInput[3] == 1:
            self.board.move_piece(1, 0)
        elif agentInput[4] == 1:
            self.board.rotate_piece(clockwise=True)
        
        self.board.drop_piece()  # Drop piece after agent manipulates it
        tetrisObject.frameNumber += 1  # Increment frameNumber, every 4 frames send stacked frames to DQN

        self.board.calcScore()
        
        # Get image data from game screen using pygame
        frame = pygame.surfarray.array3d(pygame.display.get_surface())

        # Create screen as PIL Image without GUI to reduce proccessing
        img = Image.fromarray(frame[:, :500, :])

        # Resize to neural net dimensions and convert to 8-bit black and White ('L') for reduced proccessing
        frame = img.resize((50, 50))
        frame = frame.convert(mode='L')

        imageData = numpy.asarray(frame)

        # Tick game
        rect = (0, 0, 250, 500)
        self.surface.fill((0, 0, 0), rect)
        self.board.draw()
        pygame.display.update()
        self.clock.tick(60)  # Set game speed

        # After running action see if gameOver is true
        if terminal:
            reward = -1000
            score -= 1000

        running = False

        # Pausing functionality...
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = True

        while running:
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        
        

        return imageData, reward, terminal

    def run(self):
        # Set-up variables and defaults for game...
        global level, score
        pygame.time.set_timer(Tetris.DROP_EVENT, (750 - ((level - 1) * 50)))  # Controls how often blocks drop. Each level-up takes 50ms off
        pygame.display.set_caption("Tetris V4.1")  # Set window title

def playGame(Tetris):
    Tetris.run()



# DQN Code Begins here:
# Hyperparameters
ACTIONS = 5  # Do nothing[0], rotate right[1], rotate left[2], move left[3], move right [4], drop piece to bottom[5].
INIT_EPSILON = 1  # Starting epsilon (for exploring). This will make the agent start by choosing an exploring action constantly.
FINAL_EPSILON = 0.03  # Final epsilon (final % chance to take an exploring action)
OBSERVE = 25000  # Observe game for 10000 frames. This fills the replay memory before the agent can take action
REPLAY_MEMORY = 25000  # Size of ReplayMemory
BATCH_SIZE = 32  # Size of minibatch
GAMMA = 0.02 # Decay rate of past observations

tetrisObject = Tetris()  # Create new Tetris instance

def createWeight(shape):
    #  Shape is a 1-d integer array. This defines the shape of the output tensor
    print("Creating Weight")
    weight = tf.truncated_normal(shape, stddev=0.01)  # Creates a random initial weight from a standard distribution with a standard deviation of 0.01.
    return tf.Variable(weight)


def createBias(shape):
    print("Creating bias")
    bias = tf.constant(0.01, shape=shape)
    return tf.Variable(bias)


def createConvolution(input, filter, stride):
    # Computes a convolution given 4D input tensors (input, filter)
    return tf.nn.conv2d(input, filter, strides=[1, stride, stride, 1], padding="SAME")

def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def createNetwork():
    playGame(tetrisObject)  # Start new Tetris Instance
    # Input is 80 x 80 x 4 (80 width, 80 height, 4 images stacked together)

    print("Creating Network...")

    # Create weight and bias's(?) for each layer
    layerOneWeights = createWeight([8, 8, 4, 32])  # Output Tensor is 8x8x4x32 for layer one. This will be the size of the layer 1 convolution.
    layerOneBias = createBias([32])  # Creates a constant tensor with a dimensionality of 32 (1-d)

    layerTwoWeights = createWeight([4, 4, 32, 64])  # Output tensor for layer two is 4x4x32x64. This will be the size of the layer 2 convolution.
    layerTwoBias = createBias([64])  # Creates a constant tensor with dimensionality of 64

    layerThreeWeights = createWeight([3, 3, 64, 64])  # Output Tensor is 3x3x64x64 for layer three. This will be the size of the layer 3 convolution.
    layerThreeBias = createBias([64])  # Creates a constant tensor with dimensionality of 64

    weights_fc1 = createWeight([1024, 512])  # Output tensor will be 1600 x 512. Creates weights for fully connected ReLU layer.
    bias_fc1 = createBias([512])  # Create bias for fully connected layer

    weights_fc2 = createWeight([512, ACTIONS])  # Creates weights for fullyConnectedLayer to Readout Layer.
    bias_fc2 = createBias([ACTIONS])  # Creates bias for readout layer.

    # Create layers below...
    # Create Input Layer
    # Input image is 80x80 and we feed in 4 images at once...
    input = tf.placeholder("float", [None, 50, 50, 4])  # Creates a tensor that will always be fed a tensor of floats 80x80x4 (input size)

    # The hidden layers will have a rectified linear activation function (ReLU)
    # Create first convolution (hidden layer one) by using the input and layerOneWeights and then adding the Bias
    conv1 = tf.nn.relu(createConvolution(input, layerOneWeights, stride=4) + layerOneBias)
    pool1 = max_pool(conv1)  # Perform max pooling

    # Create second convolution (hidden layer two) by using the first hidden layer (conv1) and the second layer weights. Add the second layer bias
    conv2 = tf.nn.relu(createConvolution(pool1, layerTwoWeights, stride=2) + layerTwoBias)

    # Create third and final convolution (hidden layer three) by using the second hidden layer (conv2) and the third layer weights and bias
    conv3 = tf.nn.relu(createConvolution(conv2, layerThreeWeights, stride=1) + layerThreeBias)

    # Reshape third layer convolution into a 1-d Tensor (basically a list or array)
    conv3Flat = tf.reshape(conv3, [-1, 1024])  # Use 1600 for use with weights_fc1

    hiddenFullyConnceted = tf.nn.relu(tf.matmul(conv3Flat, weights_fc1) + bias_fc1)  # Creates final hidden layer with 256 fully connected ReLU nodes

    # Create readout layer
    readout = tf.matmul(hiddenFullyConnceted, weights_fc2) + bias_fc2  # Creates readout layer

    return input, readout, hiddenFullyConnceted

def trainNetwork(inputLayer, readout, fullyConnected, sess):
    global tetrisObject, debugCounter, score, linesCleared, MAX_REWARD, gamesPlayed, FONT, startTime
    epsilon = 0
    cycleCounter = 0

    trainingTime = datetime.datetime.now()

    printOutActions = {0 : "Do Nothing", 1 : "Rotate Right", 2 : "Rotate Left", 3 : "Move Left", 4 : "Move Right", 5 : "Drop Piece"}
    # inputLayer is the inputLayer (duh), hiddenFullyConnected is the fully connected ReLU layer (second to last layer),
    # readout is the readout from the final layer (Q Values) and sess is the TensorFlow session

    print("Training Network")

    # Define cost function
    a = tf.placeholder("float", [None, ACTIONS])  # creates a float variable that will take n x ACTIONS tensors. (Used for holding actions in minibatch)
    y = tf.placeholder("float", [None])  # Creates a float tensor that will take any shape tensor as input. (used for holding yBatch in minibatch)

    readout_action = tf.reduce_sum(tf.multiply(readout, a), axis=1)  # multiples readout (Q values) by a (action Batch) and then computes the sum of the first row

    cost1 = tf.square(y - readout_action)  # Find the squared error...
    cost = tf.reduce_mean(cost1)  # Reduce

    trainingStep = tf.train.AdamOptimizer(0.001).minimize(cost)  # Creates an object for training using the ADAM algorithm with a learning rate of 0.001

    replayMemory = deque()  # Will be used to store replay memory

    # Save / load network
    saver = tf.train.Saver()  # Create new saver object for saving and restoring variables
    sess.run(tf.global_variables_initializer())  # Initialize all global variables
    savePoint = tf.train.get_checkpoint_state('savedNetworks')  # If the checkpoint file in savedNetworks directory contains a valid CheckPointState, return it.

    # If CheckPointState exists and path exists, restore it along with the replayMemory and
    if savePoint and savePoint.model_checkpoint_path:
        white = (255, 255, 255)
        black = (0, 0, 0)

        # Restore network weights
        saver.restore(sess=sess, save_path=savePoint.model_checkpoint_path)

        # Restore gameStats (score, games Played, max reward, lines cleared)
        gameParametersLoad = numpy.load('gameStatistics.npy')
        score = gameParametersLoad[0]
        linesCleared = gameParametersLoad[1]
        MAX_REWARD = gameParametersLoad[2]
        gamesPlayed = gameParametersLoad[3]
        #trainingTime += gameParametersLoad[4]

        # Update GUI
        label = FONT.render("Max Reward: " + str(MAX_REWARD), 1, white, black)
        tetrisObject.surface.blit(label, (0, 530))
        label = FONT.render("Score: " + str(score),  1, white, black)
        tetrisObject.surface.blit(label, (0,  510))
        label = FONT.render('Games Played: ' + str(gamesPlayed), 1, white, black)
        tetrisObject.surface.blit(label, (125, 530))
        label = FONT.render("Lines cleared: " + str(linesCleared), 1, white, black)
        tetrisObject.surface.blit(label, (125, 510))

        # Restore replayMemory and set network hyperparameters as if we were done observing/exploring
        replayMemory = numpy.load('replayMemory.npy')
        replayMemory = deque(replayMemory)  # Convert back into deque Object
        epsilon = FINAL_EPSILON
        cycleCounter = OBSERVE + 1
        print("Successfully restored previous session: " + savePoint.model_checkpoint_path)
    else:
        print("Could not load from save")
        cycleCounter = 0
        epsilon = INIT_EPSILON  # Set initial explore/exploit rate

    doNothing = [1, 0, 0, 0, 0, 0]  # Send do_nothing action by default for first frame

    # imageData = image data from game, reward = recieved reward, terminal = gameOver flag.
    imageData, reward, terminal = tetrisObject.handle_input(agentInput=doNothing)

    # Stack 4 copies of the first frame into a 3D array on a new axis
    frameStack = numpy.stack((imageData, imageData, imageData, imageData), axis=2)

    # Run forever - this is the main code
    while True:
        readoutEvaluated = readout.eval(feed_dict={inputLayer: [frameStack]})[0]  # readoutEvaluated is equal to the evaluation of the output layer (Q values) when feeding the input layer the newest frame

        action = numpy.zeros([ACTIONS])  # Create 1xACTIONS array (for choosing action to send)
        chosenAction = 0  # Do nothing by default

        # Explore / Exploit decision
        if random.random() <= epsilon or tetrisObject.frameNumber <= OBSERVE:  # If we should explore...
            print("Exploring!")
            # Choose action randomly
            chosenAction = random.randint(0, len(action))  # Choose random action from list of actions..
            if chosenAction == len(action):
                chosenAction = chosenAction - 1  # Prevents index out of bounds as len(action) is non-zero indexed while lists are zero-indexed
            action[chosenAction] = 1  # Set that random action to 1 (for true)
        else:
            print("Exploiting!")
            # Choose action greedily
            chosenAction = numpy.argmax(readoutEvaluated)  # Set chosenAction to the index of the largest Q-value
            action[chosenAction] = 1  # Set the largest "action" to true
            
        #print(printOutActions.get(chosenAction))  # prints the action the agent chooses at each step


        # Scale Epsilon if done observing
        if epsilon > FINAL_EPSILON and cycleCounter > OBSERVE:  # If epsilon is not final and we're done observing...
            epsilon -= 0.0002  # Subtract 0.002 from epsilon. This will reduce 2% from epsilon every 1000 timesteps...

        # Run once per frame. This will send the selected action to the game and give us our reward and then train the agent with our minibatch.
        for i in range(0, 1):
            # Run selected action and observe the reward
            frame, localScore, localTerminal = tetrisObject.handle_input(agentInput=action)  # Send selected action to game

            if localScore > 0: print("Reward: " + str(localScore) + '  Epsilon: ' + str(epsilon))
            elif localScore < -500: print("Negative Reward: " + str(localScore))

            # Reshape so that we can store each 50x50 image as a 3D numpy array
            frame = numpy.reshape(frame, (50, 50, 1))

            # Append first 3 50x50 pictures stored in framestack on new index (oldest frame = 0th index)
            frameStackNew = numpy.append(frame, frameStack[:, :, 0:3], axis=2)

            # frameStack = previous stack of frames, action = taken action, localScore = change in score (reward), frameStackNew = updated stack of frames, localTerminal = is game over?
            replayMemory.append((frameStack, action, localScore, frameStackNew, localTerminal))  # Store transition in replay memory as a tuple

            # If replay memory is full, get rid of oldest
            if len(replayMemory) > REPLAY_MEMORY:
                replayMemory.popleft()

        if cycleCounter > OBSERVE and len(replayMemory) >= BATCH_SIZE:
            # Get minibatch
            minibatch = random.sample(replayMemory, int(BATCH_SIZE))  # Use for random sampling
            
            #prioritizedReplay = []
            #prioritizedReplay2 = []
            # Get indices where lines have been cleared and high rewards that are not the first piece in a new game (start at 1000)
            #for i in range(0, len(replayMemory)):
            #    if replayMemory[i][2] > 1000:
            #        prioritizedReplay.append(replayMemory[i])  # Use for tracking indices
            #        prioritizedReplay2.append(replayMemory[i][2])  # Use for finding highest reward

            #prioritizedReplayPaddingSize = int(BATCH_SIZE - len(prioritizedReplay2))
            #print(prioritizedReplayPaddingSize)
            #largestIndices = numpy.argpartition(prioritizedReplay2, BATCH_SIZE - prioritizedReplayPaddingSize)[BATCH_SIZE - prioritizedReplayPaddingSize:]  # Get indices of top 32 rewards
            #minibatch2 = []
            
            #for x in range(0, len(largestIndices)):
            #    minibatch2.append(prioritizedReplay[largestIndices[x]])  # Append full replay to minibatch2 (rather than just the reward)
            #    print(prioritizedReplay[largestIndices[x][2]])
            
            #minibatch1 = random.sample(replayMemory, prioritizedReplayPaddingSize)  # Fill minibatch with random samples if not enough lines have been cleared
                
            #minibatch = minibatch1 + minibatch2  # Combine the minibatches so it's half random, half highest reward
            #minibatch = minibatch2  # Use for 100% prioritized experience replay.
            
            # Get batch variables
            initialFrameBatch = [r[0] for r in minibatch]
            actionBatch = [r[1] for r in minibatch]
            scoreBatch = [r[2] for r in minibatch]
            updatedFrameBatch = [r[3] for r in minibatch] # Returns
            yBatch = []  # Create blank list

            batchReadout = readout.eval(feed_dict={inputLayer: updatedFrameBatch})  # Get readout of final layer (Q Values) by feeding input layer the updated frames

            for i in range(0, len(minibatch)):
                if minibatch[i][4]:  # If game over is true for that replay, append the score for that
                    yBatch.append(scoreBatch[i])
                else:  # Otherwise append the score + GAMMA * the max value in the new Q Values for the updated frames
                    yBatch.append(scoreBatch[i] + GAMMA * numpy.max(batchReadout[i]))

            # Perform training step by feeding ADAM optimizer the actions for the scores (label)according to the corresponding 
            trainingStep.run(feed_dict={y: yBatch,
                                        a: actionBatch,
                                        inputLayer: initialFrameBatch})

        frameStack = frameStackNew  # Update Framestack
        cycleCounter += 1

        if (cycleCounter % 1000 == 0):
            trainingTime += (datetime.datetime.now() - startTime)
            print('Frame: ' + str(cycleCounter) + '  Q-Values: ' + str(readoutEvaluated))
            print('Training Time: ' + str(trainingTime))

        gameParameters = []
        # Save network every 5000 steps
        if cycleCounter % 10000 == 0:
            saver.save(sess, 'savedNetworks/Tetris-dqn', global_step=cycleCounter)  # Save to directories savedNetworks folder
            numpy.save('replayMemory', replayMemory)
            gameParameters.append(score)
            gameParameters.append(linesCleared)
            gameParameters.append(MAX_REWARD)
            gameParameters.append(gamesPlayed)
            gameParameters.append(datetime.datetime.now() - startTime)  # Save the running time for this training
            gameParameters = numpy.array(gameParameters)  # Convert to numpy array for saving
            numpy.save('gameStatistics', gameParameters)
            

if __name__ == "__main__":
    sess = tf.InteractiveSession()
    input, readout, fullyConnected = createNetwork()
    initVars = tf.global_variables_initializer()
    trainNetwork(input, readout, fullyConnected, sess)
