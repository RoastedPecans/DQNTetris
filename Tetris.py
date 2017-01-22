# Program Name: Tetris.py
# Program Purpose: Tetris built in Python for eventual use with a DNQ
# Date Started: 1/3/17
# Last Modified: 1/21/17
# Programmer: Connor

import os

#  This sets the starting position for the pygame Window. It's set to the top-left corner because that is where
#  Pyscreenshot sets its bounding box.
x = 0
y = 0
os.environ['SDL_VIDEO_WINDOW_POS'] = "%d, %d" % (x, y)

import pygame
import sys
import random
from pygame.locals import *
import numpy
import pyscreenshot as ImageGrab

score = 0
level = 1  # Start on level 1


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
        Tetris.exportFrame(self)  # This will create a new "screenshot" of the game every time the pieces drop
        if self._can_drop_piece():
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
        remove = [y for y, row in enumerate(self.board) if all(row)]
        tempScore = len(remove)
        if tempScore > 0:  # Only update score if there's a reason to
            self.score(tempScore)
        for y in remove:
            self._delete_line(y)

    def score(self, scoreToAdd):
        # Might have to change scores later on. With these scores, the agent should play to clear the most lines at the highest level
        global score
        global level
        print("Score to Add: " + str(scoreToAdd))
        if scoreToAdd == 1:
            scoreToAdd = 50 * level
        elif scoreToAdd == 2:
            scoreToAdd = 150 * level
        elif scoreToAdd == 3:
            scoreToAdd = 250 * level
        elif scoreToAdd == 4:
            scoreToAdd = 350 * level
        else:
            print("ELSE")
        score += scoreToAdd
        font = pygame.font.Font("/System/Library/Fonts/Helvetica.dfont", 24)
        white = (255, 255, 255)
        black = (0, 0, 0)
        label = font.render("Score: " + str(score),  1, white, black)
        self.surface.blit(label, (0,  530))
        print(score)
        self.levelUp()  # Every time score is updated see if we level up



    def levelUp(self):
        global level
        global score
        scoreThreshold = 1000 * level
        if score >= scoreThreshold and level < 10:  # Increment score if pass the scorethreshold and under level 10
            level += 1
            print("Level Up!")
            font = pygame.font.Font("/System/Library/Fonts/Helvetica.dfont", 24)
            white = (255, 255, 255)
            black = (0, 0, 0)
            levelLabel = font.render("Level: " + str(level), 1, white, black)
            self.surface.blit(levelLabel, (0, 500))



    def game_over(self):
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
            self.drop_piece()
        self.drop_piece()

class Tetris:
    DROP_EVENT = USEREVENT + 1
    save = False
    show = False

    def __init__(self):
        self.surface = pygame.display.set_mode((250,  550))  # Set dimensions of game window. Creates a Surface
        self.clock = pygame.time.Clock()
        self.board = Board(self.surface)
        self.exportFrame()


    def handle_key(self, event_key):
        if event_key == K_DOWN:
            self.board.drop_piece()
        elif event_key == K_LEFT:
            self.board.move_piece(dx=-1, dy=0)  # Subtract 1 from x to move left
        elif event_key == K_RIGHT:
            self.board.move_piece(dx=1, dy=0)  # Add 1 to x to move right
        elif event_key == K_UP:
            self.board.rotate_piece()
        elif event_key == K_SPACE:
            self.board.full_drop_piece()
        elif event_key == K_ESCAPE:
            self.pause()

    def pause(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == KEYDOWN and event.key == K_ESCAPE:
                    running = False

    def run(self):
        global level
        global score
        pygame.init()
        pygame.time.set_timer(Tetris.DROP_EVENT, (750 - ((level - 1) * 50)))  # Controls how often blocks drop. Each level-up takes 50ms off
        pygame.display.set_caption("Tetris V1.2")  # Set window title
        font = pygame.font.Font("/System/Library/Fonts/Helvetica.dfont", 24)
        white = (255, 255, 255)
        black = (0, 0, 0)
        label = font.render("Score: " + str(score),  1, white)
        self.surface.blit(label, (0,  530))
        levelLabel = font.render("Level: " + str(level), 1, white, black)
        self.surface.blit(levelLabel, (0, 500))

        while True:  # Gameloop
            if self.board.game_over():
                print("Game Over")
                print("Time: " + str(pygame.time.get_ticks() / 1000))   # Returns game time in seconds
                pygame.quit()
                sys.exit()
            rect = (0, 0, 250, 500)
            self.surface.fill((0, 0, 0), rect)
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == KEYDOWN:
                    self.handle_key(event.key)
                elif event.type == Tetris.DROP_EVENT:
                    self.board.drop_piece()

            self.board.draw()
            pygame.display.update()
            self.clock.tick(60)  # Set game speed

    def exportFrame(self):
        #  Gets called every time the screen is updated (when a piece drops)
        img = ImageGrab.grab(bbox=(pygame.Surface.get_bounding_rect(self.surface)))  # Screenshots the Tetris game window
        img = img.crop(box=(0, 45, 250, 547))  # Gets rid of the title bar on top of the menu. Speeds up processing.
        img = img.convert(mode='L')  # Convert to 8-bit black and white
        if Tetris.show:
            img.show()
        if Tetris.save:
            img.save("test" + (str(random.randint(0, 10000)) + ".png"), format='png')
        #frame = list(img.getdata())  # Similar to numpy.asarray. Returns all pixel data as a List.
        frame = numpy.asarray(img)  # Creates a list with all pixel values in order. This will be exported to the ML agent.
        print(frame)

if __name__ == "__main__":
    Tetris().run()
