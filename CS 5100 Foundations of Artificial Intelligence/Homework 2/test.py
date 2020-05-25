import numpy as np
import copy
import sys
NUM_COLUMNS = 8
WHITE = 1
NOBODY = 0
BLACK = -1




# f2 = open("./clearBestMove.txt", "r")
# lines = f2.readlines()
# for line3 in lines:
#     print(line3)


board = np.zeros((NUM_COLUMNS, NUM_COLUMNS))
board_chars = {
        'W': WHITE,
        'B': BLACK,
        '-': NOBODY
    }
r = 0
for line in sys.stdin:
    for c in range(NUM_COLUMNS):
        board[r][c] = board_chars.get(line[c], NOBODY)  # quietly ignore bad chars
    r += 1

print(board)

board = np.zeros((NUM_COLUMNS, NUM_COLUMNS))
board_chars = {
    'W': WHITE,
    'B': BLACK,
    '-': NOBODY
}
r = 0  # row
for line in sys.stdin:
    for c in range(NUM_COLUMNS):
        board[r][c] = board_chars.get(line[c], NOBODY)  # quietly ignore bad chars
    r += 1