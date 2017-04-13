# Program Name: Tetris.py
# Program Purpose: Tetris built in Python for use with a DQN that will learn to play
# Date Started: 1/3/17
# Last Modified: 4/1/17
# Programmer: Connor

import tensorflow as tf
from collections import deque  # Deque used for replay memory
import pygame
import numpy
import random
from PIL import Image
from copy import deepcopy

# Initialize Pygame and pygame audio
pygame.init()
pygame.mixer.init()
linesClearedSound = pygame.mixer.Sound("dingSoundEffect.wav")

# Global Variables for Tetris
score = 0  # Total score
thetaScore = 1 # Keep track of changes in score (used to reward agent)
terminal = False  # Flag for GameOver
linesCleared = 0
MAX_REWARD = 0
reward = 0
newBoard = []
gamesPlayed = 0
FONT_PATH = "C:\Windows\Fonts\Arial.ttf"  # Use C:\Windows\Fonts\Arial.ttf for Windows, /System/Library/Fonts/Helvetica.dfont for Mac.
FONT = pygame.font.Font(FONT_PATH, 12)  # Create a single object for rendering all text in game to save computation

# Parameters for DQN
ACTIONS = 5  # Do nothing[0], move left[1], rotate left[2], move left[3], move right [4].
INIT_EPSILON = 1  # Starting epsilon (for exploring). This will make the agent start by choosing a random action constantly.
FINAL_EPSILON = 0.01  # Final epsilon (final % chance to take an exploring action)
OBSERVE = 20000  # Observe game for x frames. This fills the replay memory before the agent can begin training.
REPLAY_MEMORY = OBSERVE  # Size of ReplayMemory is equal to Observe because we need to populate the replaymemory before we can train from it.
BATCH_SIZE = 64  # Size of minibatch to use in training
GAMMA = 0.95  # Decay rate of past observations. Used in Q-Learning equation and dictates whether to aim for future rewards or short-sighted rewards.
LOGGING = True  # Set to True to enable logging of when a line is cleared. timeToScore.txt will be needed in .py directory.

# Tetris game code originally from https://github.com/ktt3ja/kttetris/blob/master/tetris.py.
# Modified for use with a Deep-Q Network.

# Class for the Tetrominos
class Piece:
    # Square Tetromino
    O = (((0, 0, 0, 0, 0),  (0, 0, 0, 0, 0),  (0, 0, 1, 1, 0),  (0, 0, 1, 1, 0),  (0, 0, 0, 0, 0)), ) * 4

    # Straight Tetromino
    I = (((0, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 1, 1, 1, 1), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)),
         ((0, 0, 0, 0, 0), (0, 0, 1, 0, 0), (0, 0, 1, 0, 0), (0, 0, 1, 0, 0), (0, 0, 1, 0, 0)),
         ((0, 0, 0, 0, 0), (0, 0, 0, 0, 0), (1, 1, 1, 1, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)),
         ((0, 0, 1, 0, 0), (0, 0, 1, 0, 0), (0, 0, 1, 0, 0), (0, 0, 1, 0, 0), (0, 0, 0, 0, 0)))

    # L-Shaped Tetromino
    L = (((0, 0, 0, 0, 0), (0, 0, 1, 0, 0), (0, 0, 1, 0, 0), (0, 0, 1, 1, 0), (0, 0, 0, 0, 0)),
         ((0, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 1, 1, 1, 0), (0, 1, 0, 0, 0), (0, 0, 0, 0, 0)),
         ((0, 0, 0, 0, 0), (0, 1, 1, 0, 0), (0, 0, 1, 0, 0), (0, 0, 1, 0, 0), (0, 0, 0, 0, 0)),
         ((0, 0, 0, 0, 0), (0, 0, 0, 1, 0), (0, 1, 1, 1, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)))

    # J-Shaped Tetromino
    J = (((0, 0, 0, 0, 0), (0, 0, 1, 0, 0), (0, 0, 1, 0, 0), (0, 1, 1, 0, 0), (0, 0, 0, 0, 0)),
         ((0, 0, 0, 0, 0), (0, 1, 0, 0, 0), (0, 1, 1, 1, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)),
         ((0, 0, 0, 0, 0), (0, 0, 1, 1, 0), (0, 0, 1, 0, 0), (0, 0, 1, 0, 0), (0, 0, 0, 0, 0)),
         ((0, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 1, 1, 1, 0), (0, 0, 0, 1, 0), (0, 0, 0, 0, 0)))

    # Z-Shaped tetromino
    Z = (((0, 0, 0, 0, 0), (0, 0, 0, 1, 0), (0, 0, 1, 1, 0), (0, 0, 1, 0, 0), (0, 0, 0, 0, 0)),
         ((0, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 1, 1, 0, 0), (0, 0, 1, 1, 0), (0, 0, 0, 0, 0)),
         ((0, 0, 0, 0, 0), (0, 0, 1, 0, 0), (0, 1, 1, 0, 0), (0, 1, 0, 0, 0), (0, 0, 0, 0, 0)),
         ((0, 0, 0, 0, 0), (0, 1, 1, 0, 0), (0, 0, 1, 1, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)))

    # S-Shaped tetromino
    S = (((0, 0, 0, 0, 0), (0, 0, 1, 0, 0), (0, 0, 1, 1, 0), (0, 0, 0, 1, 0), (0, 0, 0, 0, 0)),
         ((0, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 1, 1, 0), (0, 1, 1, 0, 0), (0, 0, 0, 0, 0)),
         ((0, 0, 0, 0, 0), (0, 1, 0, 0, 0), (0, 1, 1, 0, 0), (0, 0, 1, 0, 0), (0, 0, 0, 0, 0)),
         ((0, 0, 0, 0, 0), (0, 0, 1, 1, 0), (0, 1, 1, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)))

    # T-shaped tetromino
    T = (((0, 0, 0, 0, 0), (0, 0, 1, 0, 0), (0, 0, 1, 1, 0), (0, 0, 1, 0, 0), (0, 0, 0, 0, 0)),
         ((0, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 1, 1, 1, 0), (0, 0, 1, 0, 0), (0, 0, 0, 0, 0)),
         ((0, 0, 0, 0, 0), (0, 0, 1, 0, 0), (0, 1, 1, 0, 0), (0, 0, 1, 0, 0), (0, 0, 0, 0, 0)),
         ((0, 0, 0, 0, 0), (0, 0, 1, 0, 0), (0, 1, 1, 1, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)))

    PIECES = {'O': O, 'I': I, 'L': L, 'J': J, 'Z': Z, 'S': S, 'T': T}  # Create a dictionary to hold Pieces

    def __init__(self, piece_name=None):
        if piece_name:
            self.piece_name = piece_name
        else:
            self.piece_name = random.choice(list(Piece.PIECES.keys()))  # If no "first piece" set then randomly select from a list of the keys in PIECES dictionary
        self.rotation = 0
        self.array2d = Piece.PIECES[self.piece_name][self.rotation]  # In array2d save the chosen Tetromino with the set rotation

    def __iter__(self):
        for row in self.array2d:
            yield row

    def rotate(self, clockwise=True):
        self.rotation = (self.rotation + 1) % 4 if clockwise else \
            (self.rotation - 1) % 4   # Add 1 for clockwise, subtract 1 for counterclockwise
        self.array2d = Piece.PIECES[self.piece_name][self.rotation]

# Class for the Tetris Game board
class Board:
    firstPiece = False  # Class variable to determine if this is the first piece in a game
    clearedPiece = False  # Class variable to tell if the last piece cleared a line. This prevents us from rewarding following plays that don't clear a line.

    COLLIDE_ERROR = {'no_error': 0,  'right_wall': 1, 'left_wall': 2, 'bottom': 3, 'overlap': 4}  # Dictionary for storing what kind of collision occurred

    # Create new piece
    def generatePiece(self):
        self.piece = Piece()   # Set first piece to random piece per the Piece Class init function
        self.piece_x, self.piece_y = 3, 0   # Set to center of the board

    def __init__(self, surface):
        global newBoard
        self.surface = surface
        self.width = 10   # Width of game board
        self.height = 22   # Height of game board
        self.block_size = 25   # "Block Size" of grid squares in pixels
        self.board = []
        for x in range(self.height):   # For all squares on board, set to 0 (unfilled)
            self.board.append([0] * self.width)
            newBoard.append([0] * self.width)
        self.generatePiece()  # Generate first piece

    # Called when a piece hits the bottom of the board
    def absorb_piece(self):
        for y, row in enumerate(self.piece):  # For all rows
            for x, block in enumerate(row):
                if block:
                    self.board[y+self.piece_y][x+self.piece_x] = block
        self.firstPiece = False
        self.generatePiece()  # Generate new piece

    # Tells us what kind of collision occurred
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

    # Works in tandem with _block_collide_with_board() to detect collisions
    def collide_with_board(self, dx, dy):
        for y, row in enumerate(self.piece):
            for x, block in enumerate(row):
                if block:
                    collide = self._block_collide_with_board(x=x+dx, y=y+dy)
                    if collide:
                        return collide
        return Board.COLLIDE_ERROR['no_error']

    # Tells us if a move is valid
    def _can_move_piece(self, dx, dy):
        dx_ = self.piece_x + dx
        dy_ = self.piece_y + dy
        if self.collide_with_board(dx=dx_, dy=dy_):
            return False
        return True

    # Tries to rotate piece and if it's a valid move it rotates it
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

    # Moves piece dx, dy amount
    def move_piece(self, dx, dy):
        if self._can_move_piece(dx, dy):
            self.piece_x += dx
            self.piece_y += dy

    # Tells us if we can drop a piece down a line
    def _can_drop_piece(self):
        return self._can_move_piece(dx=0,  dy=1)

    # Works with _try_rotate_piece to ensure rotations are valid before actually performing them
    def rotate_piece(self, clockwise=True):
        self._try_rotate_piece(clockwise)

    # Converts our grid blocks to pixels
    def pos_to_pixel(self, x, y):
        return self.block_size * x, self.block_size*(y-2)

    # Deletes lines sent to it by delete_lines()
    def _delete_line(self, y):
        for y in reversed(range(1, y+1)):
            self.board[y] = list(self.board[y-1])

    # Checks if a line is full and if so tells _delete_lines which line(s) to delete
    def delete_lines(self):
        global thetaScore, linesCleared, FONT, gamesPlayed

        # If all squares in a row are filled, index that row for deletion
        remove = [y for y, row in enumerate(self.board) if all(row)]

        # If there is a row to delete...
        if len(remove) > 0:
            thetaScore += len(remove) * 100  # 10000 reward per line cleared
            linesClearedSound.play()  # Play sound!
            self.clearedPiece = True
            
            # Update GUI
            linesCleared += len(remove)
            label = FONT.render("Lines cleared: " + str(linesCleared), 1, (255, 255, 255), (0, 0, 0))
            self.surface.blit(label, (125, 510))

            # Write to Scoring log if logging is enabled with the game number the line was cleared on
            if LOGGING:
                file = open('timeToScore.txt', 'a')
                file.write('\n' + str(gamesPlayed))

        # Delete lines
        for y in remove:
            self._delete_line(y)

    # Detects if there is a game over
    def game_over(self):
        return sum(self.board[0]) > 0 or sum(self.board[1]) > 0

    # Rendering for board
    def draw_blocks(self, array2d, color=(0, 0, 255),  dx=0,  dy=0):
        for y, row in enumerate(array2d):
            y += dy
            if y >= 2 and y < self.height:
                for x, block in enumerate(row):
                    if block:
                        # If there's a grid block to be drawn, draw it
                        self.delete_lines()  # See if we cleared any lines
                        x += dx
                        x_pix, y_pix = self.pos_to_pixel(x, y)
                        # draw block
                        pygame.draw.rect(self.surface, color,
                                         (  x_pix, y_pix,
                                            self.block_size,
                                            self.block_size))
                        # draw border around block
                        pygame.draw.rect(self.surface, (0,  0,  0),
                                         (  x_pix, y_pix,
                                            self.block_size,
                                            self.block_size),  1)

    # Update game screen
    def draw(self):
        self.draw_blocks(self.piece, dx=self.piece_x, dy=self.piece_y)  # Draws the piece
        self.draw_blocks(self.board)  # Redraws the whole board

    # First method that "chains" to all other methods. This drops the piece every tick.
    def drop_piece(self):
        if self.game_over():
            self.resetBoard()
        if self._can_drop_piece():
            self.move_piece(dx=0, dy=1)
        else:
            self.absorb_piece()
            self.delete_lines()

    # Reset board after a game over
    def resetBoard(self):
        print("Resetting game!")
        global terminal, newBoard, FONT, gamesPlayed, thetaScore
        terminal = True  # Set to True to punish agent in Tetris.handle_input() by reducing reward
        self.firstPiece = True  # Since it's a new game, let us know that it's the first piece
        thetaScore = 1  # Reset reward
        gamesPlayed += 1

        # Update GUI
        label = FONT.render('Games Played: ' + str(gamesPlayed), 1, (255, 255, 255), (0, 0, 0))
        self.surface.blit(label, (125, 530))
        
        # Reset boards
        self.board = []
        newBoard = []
        for x in range(self.height):
            self.board.append([0] * self.width)
            newBoard.append([0] * self.width)


class Tetris:
    rewardCounter = 3
    gameOverCounter = 3 # Set to 3 since subtraction is first
    def __init__(self):
        self.surface = pygame.display.set_mode((250, 550))  # Set dimensions of game window. Creates a Surface
        self.clock = pygame.time.Clock()  # Create game clock
        self.board = Board(self.surface)  # Create board for Tetris pieces
        pygame.display.update()

    # Method that handles the input from the DQN and returns all the neccessary parameters for the experience replay
    # Agent input is a 1 x ACTIONS list where each index represents a different possible action
    def handle_input(self, agentInput):
        pygame.event.pump()  # Needs to be called every frame so pygame can interact with OS
        global thetaScore, score, FONT, terminal, MAX_REWARD, reward

        reward = 1
        thetaScore = 1
        
        # Before running action set terminal (game over) to False so we don't get any errors
        if terminal:
            terminal = False
        
        # Do nothing[0], move left[1], rotate left[2], move left[3], move right [4].
        if agentInput[1] == 1:
            self.board.move_piece(-1, 0)
        elif agentInput[2] == 1:
            self.board.rotate_piece(clockwise=False)
        elif agentInput[3] == 1:
            self.board.move_piece(1, 0)
        elif agentInput[4] == 1:
            self.board.rotate_piece(clockwise=True)
        
        self.board.drop_piece()  # Drop piece after agent manipulates it. This runs the rest of the Tetris code.
        
        
        # Get image data from game screen using pygame
        frame = pygame.surfarray.array3d(pygame.display.get_surface())

        # Create screen as PIL Image without GUI to reduce computation
        img = Image.fromarray(frame[:, :500, :])

        # Resize to neural net dimensions (25 x 50) and convert to 8-bit black and White ('L') for reduced computation
        frame = img.resize((25, 50))
        frame = frame.convert(mode='L')

        # Save preproccessed frame as a numpy array
        imageData = numpy.asarray(frame)

        # Tick game and update game board
        rect = (0, 0, 250, 500)
        self.surface.fill((0, 0, 0), rect)
        self.board.draw()
        pygame.display.update()
        self.clock.tick(60)  # Set game speed

        # After running action see if gameOver is true
        if terminal:
            reward = -100
            score -= 100
            
        # Pausing functionality...
        running = False
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = True

        # Unpausing functionality
        while running:
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

        return imageData, reward

    def run(self):
        # Set-up variables and defaults for game...
        global score
        #pygame.time.set_timer(Tetris.DROP_EVENT, (750 - ((level - 1) * 50)))  # Controls how often blocks drop.
        pygame.display.set_caption("Tetris V4.2")  # Set window title

# Starts game given a Tetris Object
def playGame(Tetris):
    Tetris.run()

# -------------------------------------  Neural Network Code Begins Here!!! --------------------------------------------

tetrisObject = Tetris()  # Create new Tetris object

# Creates weights for the network
def createWeight(shape):
    #  Shape is a 1-d integer array. This defines the shape of the output tensor
    print("Creating Weight")
    weight = tf.truncated_normal(shape, stddev=0.01)  # Creates a random initial weight from a standard distribution with a standard deviation of 0.01.
    return tf.Variable(weight)


# Creates biases for the network
def createBias(shape):
    print("Creating bias")
    bias = tf.constant(0.01, shape=shape)
    return tf.Variable(bias)


# Creates a convolution for the network
def createConvolution(input, filter, stride):
    # Computes a convolution given 4D input tensors (input, filter)
    return tf.nn.conv2d(input, filter, strides=[1, stride, stride, 1], padding="SAME")


# Creates max pooling layers
def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# Creates network!
def createNetwork():
    playGame(tetrisObject)  # Start new Tetris Instance

    # Input is 25 x 50 x 4 (25 width, 50 height, 4 images stacked together)

    print("Creating Network...")

    # Create weight and biases for each layer
    layerOneWeights = createWeight([8, 8, 4, 32])  # Output Tensor is 8x8x4x32 for layer one. This will be the size of the layer 1 convolution.
    layerOneBias = createBias([32])

    layerTwoWeights = createWeight([4, 4, 32, 64])  # Output tensor for layer two is 4x4x32x64. This will be the size of the layer 2 convolution.
    layerTwoBias = createBias([64])

    layerThreeWeights = createWeight([3, 3, 64, 64])  # Output Tensor is 3x3x64x64 for layer three. This will be the size of the layer 3 convolution.
    layerThreeBias = createBias([64])

    weights_fc1 = createWeight([512, 512])  # Output tensor will be 512 x 512. Creates weights for fully connected ReLU layer.
    bias_fc1 = createBias([512])

    weights_fc2 = createWeight([512, ACTIONS])  # Creates weights for fullyConnectedLayer to Readout Layer.
    bias_fc2 = createBias([ACTIONS])

    # Layers created below

    # Create Input Layer
    # Input image is 25x50 and we feed in 4 images at once...
    input = tf.placeholder("float", [None, 50, 25, 4])  # Creates a tensor that will always be fed a tensor of floats 25x50x4 (input size)

    # The hidden layers will have a rectified linear activation function (ReLU)
    # Create first convolution (hidden layer one) by using the input and layerOneWeights and then adding the Bias
    conv1 = tf.nn.relu(createConvolution(input, layerOneWeights, stride=4) + layerOneBias)
    pool1 = max_pool(conv1)  # Perform max pooling

    # Create second convolution (hidden layer two) by using the first hidden layer (conv1) and the second layer weights. Add the second layer bias
    conv2 = tf.nn.relu(createConvolution(pool1, layerTwoWeights, stride=2) + layerTwoBias)

    # Create third and final convolution (hidden layer three) by using the second hidden layer (conv2) and the third layer weights and bias
    conv3 = tf.nn.relu(createConvolution(conv2, layerThreeWeights, stride=1) + layerThreeBias)

    # Reshape third layer convolution into a 1-d Tensor (basically a list or array)
    conv3Flat = tf.reshape(conv3, [-1, 512])  # Use 512 for use with weights_fc1

    hiddenFullyConnceted = tf.nn.relu(tf.matmul(conv3Flat, weights_fc1) + bias_fc1)  # Creates final hidden layer with 512 fully connected ReLU nodes

    # Create readout layer
    readout = tf.matmul(hiddenFullyConnceted, weights_fc2) + bias_fc2  # Creates readout layer

    return input, readout, hiddenFullyConnceted

# Code for training DQN and having it interact with Tetris
def trainNetwork(inputLayer, readout, fullyConnected, sess):
    global tetrisObject, score, linesCleared, MAX_REWARD, gamesPlayed, FONT
    epsilon = INIT_EPSILON
    cycleCounter = 0

    # Dictionary to hold action selected if we want to print out what action the agent chooses
    printOutActions = {0 : "Do Nothing", 1 : "Rotate Right", 2 : "Rotate Left", 3 : "Move Left", 4 : "Move Right", 5 : "Drop Piece"}

    # inputLayer is the inputLayer (duh), hiddenFullyConnected is the fully connected ReLU layer (second to last layer),
    # readout is the readout from the final layer (gives us Q Values) and sess is the TensorFlow session

    print("Training Network")

    # Define cost function
    a = tf.placeholder("float", [None, ACTIONS])  # creates a float variable that will take n x ACTIONS tensors. (Used for holding actions from minibatch)
    y = tf.placeholder("float", [None])  # Creates a float tensor that will take any shape tensor as input. (used for holding yBatch from minibatch)

    readout_action = tf.reduce_sum(tf.multiply(readout, a), axis=1)  # multiples readout (Q values) by a (action Batch) and then computes the sum of the first row

    cost1 = tf.square(y - readout_action)  # Find the squared error...
    cost = tf.reduce_mean(cost1)  # Reduce

    trainingStep = tf.train.AdamOptimizer(0.00001).minimize(cost)  # Creates an object for training using the ADAM algorithm with a learning rate of 0.001

    replayMemory = deque()  # Will be used to store experiences. This is a First-in, First-out object

    # Save / load network
    saver = tf.train.Saver()  # Create new saver object for saving and restoring variables
    sess.run(tf.global_variables_initializer())  # Initialize all global variables
    savePoint = tf.train.get_checkpoint_state('savedNetworks')  # If the checkpoint file in savedNetworks directory contains a valid CheckPointState, return it.

    # If CheckPointState exists and path exists, restore it along with the replayMemory and gameParameters
    if savePoint and savePoint.model_checkpoint_path:
        white = (255, 255, 255)
        black = (0, 0, 0)

        # Restore network weights
        saver.restore(sess=sess, save_path=savePoint.model_checkpoint_path)

        # Restore gameStats (score, games played, max reward, lines cleared)
        gameParametersLoad = numpy.load('gameStatistics.npy')
        score = gameParametersLoad[0]
        linesCleared = gameParametersLoad[1]
        MAX_REWARD = gameParametersLoad[2]
        gamesPlayed = gameParametersLoad[3]

        # Update GUI
        label = FONT.render("Max Reward: " + str(MAX_REWARD), 1, white, black)
        tetrisObject.surface.blit(label, (0, 530))
        label = FONT.render("Score: " + str(score),  1, white, black)
        tetrisObject.surface.blit(label, (0,  510))
        label = FONT.render('Games Played: ' + str(gamesPlayed), 1, white, black)
        tetrisObject.surface.blit(label, (125, 530))
        label = FONT.render('Lines Cleared: ' + str(linesCleared), 1, white, black)
        tetrisObject.surface.blit(label, (125, 510))

        # Restore replayMemory and set network hyperparameters as if we were done observing/exploring
        replayMemory = numpy.load('replayMemory.npy')
        replayMemory = deque(replayMemory)  # Convert back into deque Object
        epsilon = FINAL_EPSILON
        cycleCounter = OBSERVE + 1
        print("Successfully restored previous session: " + savePoint.model_checkpoint_path)
    else:
        print("Could not load from save")

    doNothing = [1, 0, 0, 0, 0, 0]  # Send do_nothing action by default for first frame

    # imageData = image data from game, reward = received reward
    imageData, reward = tetrisObject.handle_input(agentInput=doNothing)

    # Stack 4 copies of the first frame into a 3D array on a new axis
    frameStack = numpy.stack((imageData, imageData, imageData, imageData), axis=2)

    localScore2 = 0  # Used to ensure that we only add "good" experiences to the replay memory aka only one experience per piece

    # Run forever - this is the main code
    while True:
        readoutEvaluated = readout.eval(feed_dict={inputLayer: [frameStack]})[0]  # readoutEvaluated is equal to the evaluation of the output layer (Q values) when feeding the input layer the newest frame

        action = numpy.zeros([ACTIONS])  # Create 1xACTIONS zeros array (for choosing action to send)
        chosenAction = 0  # Do nothing by default

        # Explore / Exploit decision
        if random.random() <= epsilon:  # If we should explore...
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
            epsilon -= 0.00002  # Subtract 0.000002 from epsilon to increase exploitation chance.

        # Run once per frame. This will send the selected action to the game and give us our reward and then train the agent with our minibatch.
        for i in range(0, 1):
            # Run selected action and observe the reward
            frame, localScore = tetrisObject.handle_input(agentInput=action)  # Send selected action to game and get back the game screen and the reward receieved
            
            # Print out reward received
            #print("Reward: " + str(localScore) + '  Epsilon: ' + str(epsilon))
            
            # Reshape so that we can store each 25x50 image as a 3D numpy array  (neural network swaps dimensions)
            frame = numpy.reshape(frame, (50, 25, 1))


            # Append first 3 25x50 pictures stored in framestack on new index (oldest frame = 0th index)
            frameStackNew = numpy.append(frame, frameStack[:, :, 0:3], axis=2)

            # frameStack = previous stack of frames, action = taken action, localScore = change in score (reward), frameStackNew = updated stack of frames, localTerminal = if game over
            # Only adds experience if last piece is at bottom and new piece has not been generated yet so we don't confuse agent
            replayMemory.append((frameStack, action, localScore, frameStackNew))  # Store transition in replay memory as a tuple

            localScore2 = localScore
            
            # If replay memory is full, get rid of oldest experience
            if len(replayMemory) > REPLAY_MEMORY:
                replayMemory.popleft()

        # If we're ready to train...
        if cycleCounter > OBSERVE and len(replayMemory) >= BATCH_SIZE:
            # Sample from replayMemory randomly
            minibatch = random.sample(replayMemory, BATCH_SIZE)
            
            # Get batch variables
            initialFrameBatch = [r[0] for r in minibatch]
            actionBatch = [r[1] for r in minibatch]
            scoreBatch = [r[2] for r in minibatch]
            updatedFrameBatch = [r[3] for r in minibatch]
            yBatch = []  # Create blank list

            batchReadout = readout.eval(feed_dict={inputLayer: updatedFrameBatch})  # Get readout of final layer (Q Values) by feeding input layer the updated frames

            for i in range(0, len(minibatch)):
                # Doubled Q-Learning, 50/50 chance of taking max or min. This reduces maximization bias.
                if random.random() < 0.49:
                    yBatch.append(scoreBatch[i] + GAMMA * numpy.max(batchReadout[i]))
                else:
                    yBatch.append(scoreBatch[i] + GAMMA * numpy.min(batchReadout[i]))

            # Perform training step by feeding ADAM optimizer the actions for the scores with the respective game state
            trainingStep.run(feed_dict={y: yBatch,
                                        a: actionBatch,
                                        inputLayer: initialFrameBatch})
            
        frameStack = frameStackNew  # Update Framestack
        cycleCounter += 1

        if cycleCounter % 1000 == 0:
            file = open('maxReward.txt', 'a')
            file.write('\n' + str(MAX_REWARD) + '      ' + str(cycleCounter))

        # Uncomment to print the frame we're on and the Q-Values for the current game state
        #print('Frame: ' + str(cycleCounter) + '  Q-Values: ' + str(readoutEvaluated))

        gameParameters = []
        # Save network every 50000 steps
        if cycleCounter % 50000 == 0:
            saver.save(sess, 'savedNetworks/Tetris-dqn', global_step=cycleCounter)  # Save network weights to directory savedNetworks
            numpy.save('replayMemory', replayMemory)  # Save replay memory
            gameParameters.append(score)
            gameParameters.append(linesCleared)
            gameParameters.append(MAX_REWARD)
            gameParameters.append(gamesPlayed)
            gameParameters = numpy.array(gameParameters)  # Convert to numpy array for saving
            numpy.save('gameStatistics', gameParameters)  # Save game parameters

if __name__ == "__main__":
    sess = tf.InteractiveSession()
    input, readout, fullyConnected = createNetwork()
    initVars = tf.global_variables_initializer()
    trainNetwork(input, readout, fullyConnected, sess)
