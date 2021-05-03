import os
import sys
import numpy as np
from skimage.io import imread

GREETINGS = False
HEX_TILE_DIAM = 156
HEX_LOWER_EDGE_COORDS = [
    [135, 1696],
    [271, 1774],
    [406, 1852],
    [542, 1930],
    [676, 1852],
    [811, 1774],
    [945, 1696],
]
HEX_COL_HEIGHTS = [4, 5, 6, 7, 6, 5, 4]

def solve_3x3(state):
    # example state: "330023111"
    partial_mat = np.array([[1,1,0],[1,1,1],[0,1,1]])
    mat = np.tile(partial_mat, (3, 3))
    mat[:3, 6:] = 0
    mat[6:, :3] = 0

    moves = (-np.linalg.inv(mat).dot([int(c) for c in state])) % 4
    print(moves.reshape((3,3)))


def solve_4x4(state):
    n = 4
    mat_for_1d = np.eye(n)
    mat_for_1d[np.arange(n - 1), np.arange(1, n)] = 1
    mat_for_1d[np.arange(1, n), np.arange(n - 1)] = 1

    mat = np.zeros((n * n, n * n))
    for r, c in zip(*np.where(mat_for_1d)):
        mat[r * n:(r + 1) * n, c * n:(c + 1) * n] = mat_for_1d

    moves = (-np.linalg.inv(mat).dot([int(c) for c in state])) % 4
    print(moves.reshape((n, n)))

def parse_hex_state(image):
    cell_states = []
    image = image[:, :, 0]
    for (x, y), col_height in zip(HEX_LOWER_EDGE_COORDS, HEX_COL_HEIGHTS):
        for i in range(col_height):
            pixel_value = image[y - 25 - HEX_TILE_DIAM * i, x]
            cell_states.append(int(round((pixel_value - 17) / 13.6)))
    return ''.join(str(s) for s in cell_states)

def solve_hex(state):
    line_sizes = [4, 5, 6, 7, 6, 5, 4]
    cumulative_line_sizes = np.hstack([0, np.cumsum(line_sizes)])
    n = np.sum(line_sizes)

    mat = np.eye(n)

    connections = [] # all connections on one half

    # connections within lines
    for i in range(4 + 5 + 6 + 7):
        if i + 1 not in [4, 4 + 5, 4 + 5 + 6, 4 + 5 + 6 + 7]:
            connections.append([i, i + 1])

    # connections between lines
    for i in range(len(line_sizes) // 2):
        for j in range(cumulative_line_sizes[i], cumulative_line_sizes[i + 1]):
            connections.append([j, j + line_sizes[i]])
            connections.append([j, j + line_sizes[i] + 1])

    for r, c in connections:
        # bidirectional connection
        mat[r, c] = 1
        mat[c, r] = 1
        # mirrored positions
        mat[n - r - 1, n - c - 1] = 1
        mat[n - c - 1, n - r - 1] = 1
    state_ints = [int(c) for c in state]

    if GREETINGS:
        for idx in [4, 5, 6, 7, 11, 16, 17, 18, 19, 28, 29, 30]:
            state_ints[idx] = (state_ints[idx] + 1) % 6

    moves = np.round(np.linalg.solve(mat[:33,:33], [-i for i in state_ints][:33])).astype(int) % 6

    return ''.join(str(move) for move in moves)

script_path = os.path.dirname(__file__)
image = imread(os.path.join(script_path, "screenshot.png"))
state = parse_hex_state(image)
print(state)

presses = solve_hex(state)
print(presses)
with open(os.path.join(script_path, "presses.txt"), 'w') as f:
    f.write(presses)
