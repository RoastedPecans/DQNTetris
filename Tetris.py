# Program Name: Tetris.py
# Program Purpose: Tetris built in Python for eventual use with a DQN
# Date Started: 1/3/17
# Last Modified: 2/13/17
# Programmer: Connor

import os
import pygame
import sys
import random
from pygame.locals import *
import numpy
import pyscreenshot as ImageGrab
import tensorflow as tf
from collections import deque  # Deque used for replay memory
from PIL import Image


#  This sets the starting position for the pygame Window. It's set to the top-left corner because that is where
#  Pyscreenshot sets its bounding box by default
x = 0
y = 0
os.environ['SDL_VIDEO_WINDOW_POS'] = "%d, %d" % (x, y)

# Global Variables for game
score = 0
thetaScore = 0  # Keep track of changes in score (used to reward agent)
level = 1  # Start on level 1
frameName = 0
terminal = False  # Flag for GameOver
FONT_PATH = "/System/Library/Fonts/Helvetica.dfont"  # Set to whatever font you want the score and levels displayed as

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
        global level
        self.piece = Piece()   # Set first piece to random piece per the Piece Class init function
        self.piece_x, self.piece_y = 3, 0   # Set to center

        print(self.piece.piece_name)

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
            Tetris.reset()
        if self._can_drop_piece():
            self.move_piece(dx=0,  dy=1)
        else:
            self.absorb_piece()
            self.delete_lines()

    # Same as drop_piece but won't lag the program when using space to fully drop piece
    def drop_piece_fully(self):
        if self._can_drop_piece():
            self.move_piece(dx=0,  dy=1)
        else:
            Tetris.exportFrame(self)  # After piece is done falling, export to agent.
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
        remove = [y for y, row in enumerate(self.board) if all(row)]
        tempScore = len(remove)  # tempScore equal to the amount of lines to remove
        if tempScore > 0:  # Only update score if there's a reason to
            self.score(tempScore)
        for y in remove:
            self._delete_line(y)

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
    """
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
        global terminal
        terminal = True
        return sum(self.board[0]) > 0 or sum(self.board[1]) > 0

    def draw_blocks(self, array2d, color=(0, 0, 255),  dx=0,  dy=0):
        for y, row in enumerate(array2d):
            y += dy
            if y >= 2 and y < self.height:
                for x, block in enumerate(row):
                    if block:
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

    def clearBoard(self):
        del self.board  # Delete old board
        board = Board  # Create new board
        Tetris.board = board  # Reset to new board



class Tetris:
    DROP_EVENT = USEREVENT + 1
    save = True  # Set to true to save frame-by-frame screenshots to project directory (good for logging)
    show = False  # Set to true to display frame-by-frame screenshots (CREATES MANY WINDOWS!)
    frameNumber = 0  # Will be used to count number of frames. Used for naming and determines when to export stack to agent.
    frameStack = []  # Will be used to hold sequences of 4 frames

    def __init__(self):
        pygame.init()
        self.surface = pygame.display.set_mode((250,  550))  # Set dimensions of game window. Creates a Surface
        self.clock = pygame.time.Clock()  # Create game clock
        self.board = Board(self.surface)  # Create board for Tetris pieces

    def handle_input(self, agentInput):
        global thetaScore
        global frameName
        # Do nothing[0], rotate right[1], rotate left[2], move left[3], move right [4], drop piece to bottom[5].
        if input == 0:
            print("Do nothing")
        elif input == 1:
            self.board.rotate_piece()
        elif input == 2:
            self.board.rotate_piece(clockwise=False)
        elif input == 3:
            self.board.move_piece(-1, 0)
        elif input == 4:
            self.board.move_piece(1, 0)
        elif input == 5:
            self.board.drop_piece_fully()

        Tetris.frameNumber += 1  # Increment frameNumber, every 4 frames send stacked frames to DQN
        img = ImageGrab.grab(bbox=(0, 45, 250, 547))  # Screenshots the Tetris game window without the text at the bottom
        img = img.convert(mode='L')  # Convert to 8-bit black and white

        #img = img.resize((63, 126))  # Uncomment this line to resize output image to (63, 126). Leaving out because it only reduces file size by 174 bytes.

        if Tetris.show:  # Set flag to show images as they are created
            img.show()
        if Tetris.save:  # Set flag to save images to project directory with a random name
            fileName = "test" + str(frameName) + ".png"  # Create sequential fileName to save image
            img.save(fileName, format='png')
            frameName += 1

        #frame = list(img.getdata())  # Similar to numpy.asarray. Returns all pixel data as a List.

        frame = numpy.asarray(img)  # Creates a list with all pixel values in order. This will be exported to the ML agent.
        reward = thetaScore
        thetaScore = 0
        return frame, reward, terminal

    def run(self):
        # Set-up variables and defaults for game...
        global level
        global score
        font = pygame.font.Font(FONT_PATH, 24)
        pygame.time.set_timer(Tetris.DROP_EVENT, (750 - ((level - 1) * 50)))  # Controls how often blocks drop. Each level-up takes 50ms off
        pygame.display.set_caption("Tetris V2.01")  # Set window title
        white = (255, 255, 255)
        black = (0, 0, 0)
        label = font.render("Score: " + str(score),  1, white)
        self.surface.blit(label, (0,  530))
        levelLabel = font.render("Level: " + str(level), 1, white, black)
        self.surface.blit(levelLabel, (0, 500))
        """
        while True:  # Gameloop
            if self.board.game_over():
                print("Game Over")
                print("Time: " + str(pygame.time.get_ticks() / 1000))   # Returns game time in seconds
                Tetris.reset()
            rect = (0, 0, 250, 500)
            self.surface.fill((0, 0, 0), rect)
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == Tetris.DROP_EVENT:
                    self.board.drop_piece()
            """
    def updateGame(self):
        self.board.draw()
        pygame.display.update()
        self.clock.tick(60)  # Set game speed

    def reset(self):
        global level
        global score
        global thetaScore
        global terminal
        Board.clearBoard()  # Reset game
        level = 0
        score = 0
        thetaScore = 0
        terminal = False

    def exportFrame(self):
        #  Gets called every time the screen is updated (when a piece drops)
        global frameName
        global thetaScore
        global terminal


def playGame():
    Tetris().run()

tetris = Tetris()  # Create tetris game
tetris.run()

# DQN Code Begins here:
# Hyperparameters
ACTIONS = 6  # Do nothing[0], rotate right[1], rotate left[2], move left[3], move right [4], drop piece to bottom[5].
LEARNINGRATE = 0.01  # To start, will change at some point.
INIT_EPSILON = 1  # Starting epsilon (for exploring). This will make the agent start by choosing an exploring action constantly.
FINAL_EPSILON = 0.05  # Final epsilon (final % chance to take an exploring action)
OBSERVE = 10  # Observe game for 50000 frames before doing anything. This fills the replay memory before the agent can take action
REPLAY_MEMORY = 50000
BATCH_SIZE = 32  # Size of minibatch
GAMMA = 0.99  # Decay rate of past observations


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
    # Computes a 2D convolution given 4D input tensors (input, filter)
    return tf.nn.conv2d(input, filter, strides=[1, stride, stride, 1], padding="SAME")


def createNetwork():
    # Input is 250 x 502 x 2 x 3 (250 width, 502 height, 8-bit image, 3 images stacked together)
    # A Tensor is an n-dimensional array
    print("Creating Network...")

    # Create weight and bias's(?) for each layer
    layerOneWeights = createWeight([8, 8, 4, 32])  # Output Tensor is 8x8x4x32 for layer one. This will be the size of the layer 1 convolution.
    layerOneBias = createBias([32])  # Creates a constant tensor with a dimensionality of 32 (1-d)

    layerTwoWeights = createWeight([4, 4, 32, 64])  # Output tensor for layer two is 4x4x32x64. This will be the size of the layer 2 convolution.
    layerTwoBias = createBias([64])  # Creates a constant tensor with dimensionality of 64

    layerThreeWeights = createWeight([3, 3, 64, 64])  # Output Tensor is 3x3x64x64 for layer three. This will be the size of the layer 3 convolution.
    layerThreeBias = createBias([64]) # Creates a constant tensor with dimensionality of 64

    # 576's below were original set to 512. use 576 for 3 x 3 x 64
    weights_fc1 = createWeight([576, 576])  # Output tensor will be 1600 x ACTIONS. Creates weights for fully connected ReLU layer.
    bias_fc1 = createBias([576])  # Tensor will be equal to the amount of actions. Creates bias for fully connected ReLU layer

    weights_fc2 = createWeight([576, ACTIONS])  # Output tensor will be 4x4 (In this case). Creates weights for fullyConnectedLayer to Readout Layer.
    bias_fc2 = createBias([ACTIONS])  # Tensor will be 4. Creates bias for readout layer.

    # Create layers below...

    # Create Input Layer
    # Input image is 250x502 and we feed in 4 images at once...
    input = tf.placeholder("float", [None, 250, 502, 4])  # Creates a tensor that will always be fed a tensor of floats 250x502x4 (input size)

    # The hidden layers will have a rectified linear activation function (ReLU)
    # Create first convolution (hidden layer one) by using the input and layerOneWeights and then adding the Bias
    conv1 = tf.nn.relu(createConvolution(input, layerOneWeights, stride=4) + layerOneBias)

    # Create second convolution (hidden layer two) by using the first hidden layer (conv1) and the second layer weights. Add the second layer bias
    conv2 = tf.nn.relu(createConvolution(conv1, layerTwoWeights, stride=2) + layerTwoBias)

    # Create third and final convolution (hidden layer three) by using the second hidden layer (conv2) and the third layer weights and bias
    conv3 = tf.nn.relu(createConvolution(conv2, layerThreeWeights, stride=1) + layerThreeBias)

    # Reshape third layer convolution into a 1-d Tensor (basically a list or array)
    conv3Flat = tf.reshape(conv3, [-1, 576])  # Use 1600 for use with weights_fc1

    hiddenFullyConnceted = tf.nn.relu(tf.matmul(conv3Flat, weights_fc1) + bias_fc1)  # Creates final hidden layer with 256 fully connected ReLU nodes

    # Create readout layer
    readout = tf.matmul(hiddenFullyConnceted, weights_fc2) + bias_fc2  # Creates readout layer (3x1?)

    return input, readout, hiddenFullyConnceted


def trainNetwork(input, readout, fullyConnected, sess):
    global tetris
    # input is the pixel input from the game, hiddenFullyConnected is the fully connected ReLU layer (second to last layer),
    # readout is the readout from the final layer (action to take) and sess is the TensorFlow session
    print("Training Network")

    # Create cost function
    a = tf.placeholder("float", [None, ACTIONS])  # creates a float variable that will take n x ACTIONS tensors
    y = tf.placeholder("float", [None])  # Creates a float tensor that will take any shape tensor as input

    readout_action = tf.reduce_sum(tf.mul(readout, a), axis=1)  # multiples readout (Q values) by a (n x ACTIONS) and then computes the sum of the first row
    cost = tf.reduce_mean(tf.square(y - readout_action))  # Computes the squared values of y - readout_action and then finds the mean of all the elements since no axis is provided
    trainingStep = tf.train.AdamOptimizer(1e-6).minimize(cost)  # Calculates the training step to take by minimizing the cost with 1e-6 as a learning rate according to ADAM algorithm...

    replayMemory = deque()  # Will be used to store replay memory

    # Save / load network
    saver = tf.train.Saver()  # Create new saver object for saving and restoring variables
    sess.run(tf.global_variables_initializer())  # Initialize all global variables
    savePoint = tf.train.get_checkpoint_state('savedNetworks')  # If the checkpoint file in savedNetworks directory contains a valid CheckPointState, return it.

    # If CheckPointState exists and path exists, restore it
    if savePoint and savePoint.model_checkpoint_path:
        saver.restore(sess=sess, save_path=savePoint.model_checkpoint_path)
        print("Successfully restored: " + savePoint.model_checkpoint_path)
    else:
        print("Could not load from save")


    doNothing = numpy.zeros(ACTIONS)  # Create a zeros array for the actions
    doNothing[0] = 1  # Send do_nothing by default

    epsilon = INIT_EPSILON

    imageData, reward, terminal = tetris.handle_input(agentInput=doNothing)   # localInput = image data from game, reward = reward, terminal = gameOver flag.
    frameStack = numpy.stack((imageData, imageData, imageData, imageData), axis=0)  # Create inital stack of images for feeding. We will append new frames to this.
    print(frameStack.shape)
    cycleCounter = 0  # Used to count frames
    # Run forever
    while True:
        # the output layer will contain the Q (action) values for each action the agent can perform (hence layer size 6)

        readoutEvaluated = readout.eval(feed_dict={input: [frameStack]})[0]  # readout_local is equal to the evaluation of the output layer when feeding the input layer the newest frame
        action = numpy.zeros([ACTIONS])
        chosenAction = 0

        print('test' + str(readoutEvaluated))

        # Explore / Exploit decision
        if random.random() <= epsilon or tetris.frameNumber <= OBSERVE:  # If we should explore...
                # Choose action randomly
                chosenAction = random.randint(0, len(action))  # Choose random action from list of actions..
                print("Chosen action: " + str(chosenAction))
                if chosenAction == len(action):
                     chosenAction = chosenAction - 1  # Prevents index out of bounds as len(action) is non-zero indexed while lists are zero-indexed
                action[chosenAction] = 1  # Set that random action to 1 (for true)
        else:
                # Choose action greedily
                print (readoutEvaluated)
                chosenAction = action.argmax(readoutEvaluated)  # Set action to the largest value for the output layer when fed the input
                action[chosenAction] = 1  # Set the largest "action" to true

        # Scale Epsilon if done observing
        if epsilon > FINAL_EPSILON and cycleCounter > OBSERVE:  # If epsilon is not final and we're done observing...
            epsilon -= 0.002  # Subtract 0.002 from epsilon. This will reduce 2% from epsilon every 1000 timesteps...

        # Run once per frame. This will populate Q values for each state and allow us to fill the replay memory
        for i in range(0, 1):
            # Run selected action and observe the reward
            frame, localScore, localTerminal = tetris.handle_input(agentInput=action)  # Send selected action to game

            #frame, localScore, localTerminal = tetris.exportFrame()  # Should we be stacking the images?

            frameStackNew = numpy.append(frame, frameStack[:, :, 0:3], axis=0)  # Append framestack to new frame

            # frameStack = previous stack of frames, action = taken action, localScore = change in score (reward), frameStackNew = updated stack of frames, localTerminal = is game over?
            replayMemory.append((frameStack, action, localScore, frameStackNew, localTerminal))  # Store transition in replay memory

            # If replay memory is full, get rid of oldest
            if len(replayMemory) > REPLAY_MEMORY:
                replayMemory.popleft()

        if cycleCounter > OBSERVE:
            # Sample miniBatch
            minibatch = random.sample(replayMemory, BATCH_SIZE)  # Get BATCH_SIZE random samples from replayMemory

            # Get batch variables
            initialFrameBatch = [replayMemory[0] for r in minibatch]
            actionBatch = [replayMemory[1] for a in minibatch]
            scoreBatch = [replayMemory[2] for s in minibatch]
            updatedFrameBatch = [replayMemory[3] for r in minibatch]

            yBatch = []  # Create blank list

            batchReadout = readout.eval(feed_dict={input : updatedFrameBatch})  # Get readout of final layer (Q Values) by feeding input layer the updated frames

            for i in range(0, len(minibatch)):
                if minibatch[i][4]:
                    yBatch.append(scoreBatch[i])
                else:
                    yBatch.append(scoreBatch[i] + GAMMA * numpy.max(batchReadout[i]))

            # Perform gradient step by feeding trainingStep
            trainingStep.run(feed_dict={y : yBatch,
                                        a : actionBatch,
                                        input : initialFrameBatch})

        frameStack = frameStackNew  # Update Framestack
        cycleCounter += 1

        # Save network every 5000 steps
        if cycleCounter % 5000 == 0:
            saver.save(sess, 'savedNetworks/Tetris-dqn', global_step=cycleCounter)

        tetris.updateGame()

if __name__ == "__main__":
    sess = tf.InteractiveSession()
    input, readout, fullyConnected = createNetwork()
    initVars = tf.global_variables_initializer()
    trainNetwork(input, readout, fullyConnected, sess)
