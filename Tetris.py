# Program Name: Tetris.py
# Program Purpose: Tetris built in Python for use with a DQN that will learn to play
# Date Started: 1/3/17
# Last Modified: 2/16/17
# Programmer: Connor

import tensorflow as tf
from collections import deque  # Deque used for replay memory
import pygame
import numpy
from pygame.locals import *

import random
from PIL import Image
import csv
import time

pygame.init()

# Global Variables for game
score = 0  # Total score
thetaScore = 0  # Keep track of changes in score (used to reward agent)
thetaScore2 = 0  # Used with thetaScore to get reward per frame
level = 1  # Start on level 1
frameName = 0  # Used for naming screenshots
terminal = False  # Flag for GameOver
linesCleared = 0
FONT_PATH = "/System/Library/Fonts/Helvetica.dfont"  # Use C:\Windows\Fonts\Arial.ttf for Windows, /System/Library/Fonts/Helvetica.dfont for Mac.
FONT = pygame.font.Font(FONT_PATH, 12)

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
    COLLIDE_ERROR = {'no_error': 0,  'right_wall': 1, 'left_wall': 2, 'bottom': 3, 'overlap': 4}  # Dictionary for storing what kind of collision occurred

    def generate_piece(self):
        thetaScore = thetaScore2 = 0
        self.piece = Piece()   # Set first piece to random piece per the Piece Class init function
        self.piece_x, self.piece_y = 3, 0   # Set to center

    def __init__(self, surface):
        self.surface = surface
        self.width = 10   # Width of game board
        self.height = 22   # Height of game board
        self.block_size = 25   # "Block Size" of grid squares in pixels
        self.board = []
        for x in range(self.height):   # For all "points" on board, set to 0
            self.board.append([0] * self.width)
        self.generate_piece()   # Generate first piece
        self.font = pygame.font.Font(FONT_PATH, 24)

    def absorb_piece(self):
        for y, row in enumerate(self.piece):  # For all rows
            for x, block in enumerate(row):
                if block:
                    self.board[y+self.piece_y][x+self.piece_x] = block
        self.generate_piece()

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

    def drop_piece(self):
        if self.game_over():
            self.resetBoard()  # If game over, reset board
        if self._can_drop_piece():
            self.move_piece(dx=0,  dy=1)
        else:
            self.absorb_piece()
            self.delete_lines()

    # Same as drop_piece but won't lag the program when using space to fully drop piece
    def drop_piece_fully(self):
        while self._can_drop_piece():
            self.move_piece(dx=0,  dy=1)
        else:
            self.absorb_piece()
            self.delete_lines()

    def rotate_piece(self, clockwise=True):
        self._try_rotate_piece(clockwise)

    def pos_to_pixel(self, x ,y):
        return self.block_size* x, self.block_size*(y-2)

    def _delete_line(self, y):
        for y in reversed(range(1, y+1)):  # Start by clearing top row first
            self.board[y] = list(self.board[y-1])

    def delete_lines(self):
        global thetaScore
        global linesCleared
        global FONT
        remove = [y for y, row in enumerate(self.board) if all(row)]
        if len(remove) > 0:
            thetaScore += len(remove) * 1000  # 1000 reward per line cleared
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

    def score(self):
        global thetaScore
        for y, row in enumerate(self.board):
            if y > 12:
                thetaScore += sum(row) * (y - 12)  # Give agent reward of 1 for every block in their line x the y (lower lines are higher y's)

    def game_over(self):
        return sum(self.board[0]) > 0 or sum(self.board[1]) > 0

    def draw_blocks(self, array2d, color=(0, 0, 255),  dx=0,  dy=0):
        global thetaScore
        global thetaScore2
        localScore = 0
        for y, row in enumerate(array2d):
            y += dy
            thetaScore2 = thetaScore
            localScore += (sum(row) * y)  # Give score for every block in row times the row height (lower is bigger)
            thetaScore += localScore - thetaScore2
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

    def full_drop_piece(self):
        while self._can_drop_piece():
            self.drop_piece_fully()
        self.drop_piece()

    def resetBoard(self):
        #print("Resetting game!")
        global level
        global thetaScore
        global terminal
        terminal = True
        level = 0
        thetaScore -= 5000

        # Reset board
        self.board = []
        for x in range(self.height):
            self.board.append([0] * self.width)


class Tetris:
    DROP_EVENT = USEREVENT + 1
    save = False  # Set to true to save frame-by-frame screenshots to project directory (good for logging)
    show = False  # Set to true to display frame-by-frame screenshots (CREATES MANY WINDOWS!)
    frameNumber = 0  # Will be used to count number of frames.
    frameStack = []  # Will be used to hold sequences of 4 frames

    def __init__(self):
        self.surface = pygame.display.set_mode((250,  550))  # Set dimensions of game window. Creates a Surface
        self.clock = pygame.time.Clock()  # Create game clock
        self.board = Board(self.surface)  # Create board for Tetris pieces
        pygame.display.update()

    def handle_input(self, agentInput):
        pygame.event.pump()  # Needs to be called every frame so pygame can interact with OS
        global thetaScore
        global thetaScore2
        global frameName
        global score
        global FONT
        global MAX_REWARD
        global terminal

        if terminal:
            terminal = False

        # Do nothing[0], rotate right[1], rotate left[2], move left[3], move right [4], drop piece to bottom[5].
        if agentInput == 1:
            self.board.rotate_piece()
        elif agentInput == 2:
            self.board.rotate_piece(clockwise=False)
        elif agentInput == 3:
            self.board.move_piece(-1, 0)
        elif agentInput == 4:
            self.board.move_piece(1, 0)
        elif agentInput == 5:
            self.board.drop_piece_fully()

        self.board.drop_piece()  # Drop piece after agent manipulates it
        tetrisObject.frameNumber += 1  # Increment frameNumber, every 4 frames send stacked frames to DQN

        # Get image data as string, convert to image, resize image, convert to 8-bit black and white.
        img = pygame.image.tostring(self.surface, 'RGB')
        img3 = Image.frombytes('RGB', (250, 502), img)
        img3 = img3.resize([80, 80])
        img3 = img3.convert(mode='L')

        frame = numpy.asarray(img3)  # Create numpy array of pixel data

        if tetrisObject.show:  # Set flag to show images as they are created
            img3.show()
        if tetrisObject.save:  # Set flag to save images to project directory with a random name
            fileName = "test" + str(frameName) + ".png"  # Create sequential fileName to save image
            img3.save(fileName, format='png')
            frameName += 1

        thetaScore = round(thetaScore)
        reward = thetaScore # Reward to return for this action

        white = (255, 255, 255)
        black = (0, 0, 0)
        
        if reward > MAX_REWARD:
            MAX_REWARD = reward
            label = FONT.render("Max Reward: " + str(MAX_REWARD), 1, white, black)
            self.surface.blit(label, (0, 530))
            
        score += round(reward / 10) # Track total reward. Divide by 10 to prevent huge score from being displayed in GUI

        # Update GUI to show total reward so far
        label = FONT.render("Score: " + str(score),  1, white, black)
        self.surface.blit(label, (0,  510))


        # Tick game
        rect = (0, 0, 250, 500)
        self.surface.fill((0, 0, 0), rect)
        self.board.draw()
        pygame.display.update()
        self.clock.tick(60)  # Set game speed

        return frame, reward, terminal

    def run(self):
        # Set-up variables and defaults for game...
        global level
        print("Run")
        global score
        global FONT
        pygame.time.set_timer(Tetris.DROP_EVENT, (750 - ((level - 1) * 50)))  # Controls how often blocks drop. Each level-up takes 50ms off
        pygame.display.set_caption("Tetris V3.4")  # Set window title
        white = (255, 255, 255)
        label = FONT.render("Score: " + str(score),  1, white)
        self.surface.blit(label, (0,  510))
        label2 = FONT.render("Lines cleared: " + str(linesCleared), 1, white)
        self.surface.blit(label2, (125, 510))
        #levelLabel = font.render("Level: " + str(level), 1, white, black)
        #self.surface.blit(levelLabel, (0, 500))


def playGame(Tetris):
    Tetris.run()



# DQN Code Begins here:
# Hyperparameters
ACTIONS = 6  # Do nothing[0], rotate right[1], rotate left[2], move left[3], move right [4], drop piece to bottom[5].
INIT_EPSILON = 1  # Starting epsilon (for exploring). This will make the agent start by choosing an exploring action constantly.
FINAL_EPSILON = 0.05  # Final epsilon (final % chance to take an exploring action)
OBSERVE = 10000  # Observe game for 10000 frames. This fills the replay memory before the agent can take action
REPLAY_MEMORY = 10000  # Size of ReplayMemory
BATCH_SIZE = 32  # Size of minibatch
GAMMA = 0.99  # Decay rate of past observations
MAX_REWARD = 0 # Max reward received 

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

    weights_fc1 = createWeight([1600, 512])  # Output tensor will be 1600 x 512. Creates weights for fully connected ReLU layer.
    bias_fc1 = createBias([512])  # Create bias for fully connected layer

    weights_fc2 = createWeight([512, ACTIONS])  # Creates weights for fullyConnectedLayer to Readout Layer.
    bias_fc2 = createBias([ACTIONS])  # Creates bias for readout layer.

    # Create layers below...
    # Create Input Layer
    # Input image is 80x80 and we feed in 4 images at once...
    input = tf.placeholder("float", [None, 80, 80, 4])  # Creates a tensor that will always be fed a tensor of floats 80x80x4 (input size)

    # The hidden layers will have a rectified linear activation function (ReLU)
    # Create first convolution (hidden layer one) by using the input and layerOneWeights and then adding the Bias
    conv1 = tf.nn.relu(createConvolution(input, layerOneWeights, stride=4) + layerOneBias)
    pool1 = max_pool(conv1)  # Perform max pooling

    # Create second convolution (hidden layer two) by using the first hidden layer (conv1) and the second layer weights. Add the second layer bias
    conv2 = tf.nn.relu(createConvolution(pool1, layerTwoWeights, stride=2) + layerTwoBias)

    # Create third and final convolution (hidden layer three) by using the second hidden layer (conv2) and the third layer weights and bias
    conv3 = tf.nn.relu(createConvolution(conv2, layerThreeWeights, stride=1) + layerThreeBias)

    # Reshape third layer convolution into a 1-d Tensor (basically a list or array)
    conv3Flat = tf.reshape(conv3, [-1, 1600])  # Use 1600 for use with weights_fc1

    hiddenFullyConnceted = tf.nn.relu(tf.matmul(conv3Flat, weights_fc1) + bias_fc1)  # Creates final hidden layer with 256 fully connected ReLU nodes

    # Create readout layer
    readout = tf.matmul(hiddenFullyConnceted, weights_fc2) + bias_fc2  # Creates readout layer

    return input, readout, hiddenFullyConnceted


def trainNetwork(inputLayer, readout, fullyConnected, sess):
    global tetrisObject
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

    # If CheckPointState exists and path exists, restore it
    if savePoint and savePoint.model_checkpoint_path:
        saver.restore(sess=sess, save_path=savePoint.model_checkpoint_path)
        print("Successfully restored: " + savePoint.model_checkpoint_path)
        epsilon = FINAL_EPSILON
    else:
        print("Could not load from save")

    doNothing = 0  # Send do_nothing action by default

    epsilon = INIT_EPSILON

    imageData, reward, terminal = tetrisObject.handle_input(agentInput=doNothing)   # imageData = image data from game, reward = recieved reward, terminal = gameOver flag.

    frameStack = numpy.stack((imageData, imageData, imageData, imageData), axis=2)  # Create inital stack of images for feeding. We will append new frames to this.
    cycleCounter = 0  # Used to count frames

    # Run forever - this is the main code
    while True:

        readoutEvaluated = readout.eval(feed_dict={inputLayer: [frameStack]})[0]  # readoutEvaluated is equal to the evaluation of the output layer (Q values) when feeding the input layer the newest frame

        action = numpy.zeros([ACTIONS])  # Create 1xACTIONS array (for choosing action to send)
        chosenAction = 0  # Do nothing by default

        #print(str(readoutEvaluated))  # Prints Q-Values at each step

        # Explore / Exploit decision
        if random.random() <= epsilon or tetrisObject.frameNumber <= OBSERVE:  # If we should explore...
            #print("Exploring!")
            # Choose action randomly
            chosenAction = random.randint(0, len(action))  # Choose random action from list of actions..
            if chosenAction == len(action):
                chosenAction = chosenAction - 1  # Prevents index out of bounds as len(action) is non-zero indexed while lists are zero-indexed
            action[chosenAction] = 1  # Set that random action to 1 (for true)
        else:
            #print("Exploiting!")
            # Choose action greedily
            chosenAction = numpy.argmax(readoutEvaluated)  # Set chosenAction to the index of the largest Q-value
            action[chosenAction] = 1  # Set the largest "action" to true

        #print(printOutActions.get(chosenAction))  # prints the action the agent chooses at each step


        # Scale Epsilon if done observing
        if epsilon > FINAL_EPSILON and cycleCounter > OBSERVE:  # If epsilon is not final and we're done observing...
            epsilon -= 0.002  # Subtract 0.002 from epsilon. This will reduce 2% from epsilon every 1000 timesteps...

        # Run once per frame. This will send the selected action to the game and give us our reward and then train the agent with our minibatch.
        for i in range(0, 1):
            # Run selected action and observe the reward
            actionToSend = action.argmax(axis=0)  # Get index of largest element...
            #print("Action selected: " + printOutActions.get(action))
            frame, localScore, localTerminal = tetrisObject.handle_input(agentInput=actionToSend)  # Send selected action to game
            #print("Reward: " + str(localScore))
            frame = numpy.reshape(frame, (80, 80, 1))  # Add an extra dimension so that we can append it to the framestack

            frameStackNew = numpy.append(frame, frameStack[:, :, 0:3], axis=2)  # Append framestack to new frame
            # frameStack = previous stack of frames, action = taken action, localScore = change in score (reward), frameStackNew = updated stack of frames, localTerminal = is game over?
            replayMemory.append((frameStack, action, localScore, frameStackNew, localTerminal))  # Store transition in replay memory as a tuple

            # If replay memory is full, get rid of oldest
            if len(replayMemory) > REPLAY_MEMORY:
                replayMemory.popleft()

        if cycleCounter > OBSERVE and len(replayMemory) >= BATCH_SIZE:
            # Sample miniBatch
            minibatch = random.sample(replayMemory, BATCH_SIZE)  # Get BATCH_SIZE random samples from replayMemory

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

            # Perform training step by feeding trainingStep
            trainingStep.run(feed_dict={y: yBatch,
                                        a: actionBatch,
                                        inputLayer: initialFrameBatch})


        frameStack = frameStackNew  # Update Framestack
        cycleCounter += 1


        # Save network every 5000 steps
        if cycleCounter % 5000 == 0:
            saver.save(sess, 'savedNetworks/Tetris-dqn', global_step=cycleCounter)  # Save to directories savedNetworks folder

if __name__ == "__main__":
    sess = tf.InteractiveSession()
    input, readout, fullyConnected = createNetwork()
    initVars = tf.global_variables_initializer()
    trainNetwork(input, readout, fullyConnected, sess)
