import os
import sys
import glob
import time
from copy import deepcopy
import numpy as np
from skimage.io import imread, imsave
from skimage.feature import peak_local_max, match_template
from skimage.transform import resize

DEBUG = True
TEMPLATE_UNIT_SIZE = 31.9
SCRIPT_DIR = os.path.dirname(__file__)
FORCE_SAVE = False
global MAX_DEPTH
MAX_DEPTH = 0
INTERACTIVE = False


DIGIT_TEMPLATES = []
for i in range(10):
    raw_image = imread(os.path.join(SCRIPT_DIR, "number_templates", f'{i}.png'))
    DIGIT_TEMPLATES.append(raw_image[:, :, :3].mean(axis=2))


def line_state_str(line_state, square_aspect=True):
    state_map = {-1: '?', 0: '@', 1: ' '}
    block_width = 2 if square_aspect else 1
    return ''.join(state_map.get(s, str(s)) * block_width for s in line_state)


def puzzle_state_str(puzzle_state, labels=False):
    if puzzle_state.shape[1] < 10:
        state_str = '    ' + ''.join(f'{i + 1:2d}' for i in range(puzzle_state.shape[1])) + '\n'
    else:
        state_str = '    ' + ''.join(f'  {i + 1:2d}' for i in range(1, puzzle_state.shape[1], 2)) + '\n'
        state_str += '    ' + ''.join(f'{i + 1:2d}  ' for i in range(0, puzzle_state.shape[1], 2)) + '\n'
        
    state_str += '   ' + '*' * (puzzle_state.shape[1] * 2 + 1) + '\n'
    state_str += '\n'.join(f'{i + 1:2d} *' + line_state_str(puzzle_state[i, :]) for i in range(puzzle_state.shape[0]))
    return state_str


def crop(arr, region):
    slices = tuple([slice(*dim_range) for dim_range in region] + [...])
    return arr[slices]


def get_puzzle_regions(image, color):
    brightness = image.mean(axis=2)
    white_mask = brightness > 200
    imsave(os.path.join(SCRIPT_DIR, "white_mask.png"), white_mask.astype(np.uint8) * 255)
    puzzle_region = [[0, image.shape[0]], [0, image.shape[1]]]

    # remove fixed size UI at top and bottom
    puzzle_region[0][0] += 136
    if color:
        puzzle_region[0][1] -= 303
    else:
        puzzle_region[0][1] -= 207

    # find white region surrounding puzzle
    center = [(puzzle_region[0][0] + puzzle_region[0][1]) // 2, (puzzle_region[1][0] + puzzle_region[1][1]) // 2]
    ul_region = [
        [puzzle_region[0][0], puzzle_region[0][0] + center[0]],
        [puzzle_region[1][0], puzzle_region[1][0] + center[1]],
    ]
    lr_region = [
        [puzzle_region[0][0] + center[0], puzzle_region[0][1]],
        [puzzle_region[1][0] + center[1], puzzle_region[1][1]],
    ]

    ul_nonwhite_mask = np.pad(
        ~crop(white_mask, ul_region),
        ((1, 0), (1, 0)),
        constant_values=True,
    )
    start_row = np.argmax(np.argmax(ul_nonwhite_mask[:, ::-1], axis=1)) - 1
    start_col = np.argmax(np.argmax(ul_nonwhite_mask[::-1, :], axis=0)) - 1

    lr_height = lr_region[0][1] - lr_region[0][0]
    lr_width = lr_region[1][1] - lr_region[1][0]
    lr_nonwhite_mask = np.pad(
        ~crop(white_mask, lr_region),
        ((0, 1), (0, 1)),
        constant_values=True,
    )
    end_row = lr_height - np.argmax(np.argmax(lr_nonwhite_mask, axis=1)[::-1]) + center[0]
    end_col = lr_width - np.argmax(np.argmax(lr_nonwhite_mask, axis=0)[::-1]) + center[1]

    puzzle_region = [
        [puzzle_region[0][0] + start_row, puzzle_region[0][0] + end_row],
        [puzzle_region[1][0] + start_col, puzzle_region[1][0] + end_col],
    ]

    # find puzzle in white region
    puzzle_rows = np.nonzero(np.any(~crop(white_mask, puzzle_region), axis=1))[0]
    puzzle_cols = np.nonzero(np.any(~crop(white_mask, puzzle_region), axis=0))[0]
    assert len(puzzle_rows) >= 2
    assert len(puzzle_cols) >= 2
    puzzle_region[0] = [puzzle_region[0][0] + puzzle_rows[0] + 1, puzzle_region[0][0] + puzzle_rows[-1] + 1]
    puzzle_region[1] = [puzzle_region[1][0] + puzzle_cols[0] + 1, puzzle_region[1][0] + puzzle_cols[-1] + 1]

    # find row and column number regions
    row_transitions = np.diff((np.mean(crop(white_mask, puzzle_region), axis=1) > 0.005).astype(np.int8))
    col_transitions = np.diff((np.mean(crop(white_mask, puzzle_region), axis=0) > 0.005).astype(np.int8))
    # move past noise on top or left edges
    row_transitions[:5] = 0
    col_transitions[:5] = 0
    # first transitions from lines with >=99.5% dark to lines with <99.5%
    starting_row = np.nonzero(row_transitions == -1)[0][0]  # row separating numbers from puzzle
    starting_col = np.nonzero(col_transitions == -1)[0][0]

    row_numbers_region = [
        [puzzle_region[0][0] + starting_row + 1, puzzle_region[0][1]],
        [puzzle_region[1][0], puzzle_region[1][0] + starting_col + 2],
    ]
    col_numbers_region = [
        [puzzle_region[0][0], puzzle_region[0][0] + starting_row + 2],
        [puzzle_region[1][0] + starting_col + 1, puzzle_region[1][1]],
    ]
    grid_region = [
        [puzzle_region[0][0] + starting_row + 1, puzzle_region[0][1]],
        [puzzle_region[1][0] + starting_col + 1, puzzle_region[1][1]],
    ]
    return {'puzzle': puzzle_region, 'row_numbers': row_numbers_region, 'col_numbers': col_numbers_region, 'grid': grid_region}

def get_gridline_indices(grid_crop, axis):
    other_axis = 1 if axis == 0 else 0
    dx = np.median(np.diff(grid_crop, axis=axis), axis=other_axis)
    large_dx_indices = np.argwhere(dx > 10).flatten()

    line_indices = [large_dx_indices[0]]
    unit_sizes = []
    for index in large_dx_indices[1:]:
        if index - line_indices[-1] <= 2:
            if dx[index] > dx[line_indices[-1]]:
                line_indices[-1] = index
        else:
            line_indices.append(index)
            unit_sizes.append(line_indices[-1] - line_indices[-2])
    return line_indices, np.mean(unit_sizes)

def parse_number(number_crop, unit_size):
    corr_thresh = 0.75
    crop_gs = number_crop[:, :, :3].mean(axis=2, keepdims=False)
    new_size = np.round(np.array(crop_gs.shape) * TEMPLATE_UNIT_SIZE / unit_size).astype(int)
    resized = np.squeeze(resize(crop_gs, new_size))
    digit_finds = []
    for i in range(10):
        match_response = match_template(resized, DIGIT_TEMPLATES[i], pad_input=True, mode='maximum')
        match_response[:12, :] = 0
        match_response[22:, :] = 0
        coords = peak_local_max(np.abs(match_response), min_distance=3)
        peak_responses = np.abs(match_response[coords[:, 0], coords[:, 1]])
        for j in range(peak_responses.shape[0]):
            if peak_responses[j] > corr_thresh:
                digit_finds.append((i, coords[j, :], peak_responses[j]))
    
    digit_finds = sorted(digit_finds, key=lambda x: x[1][1])

    last_corr = 0
    last_col = 0
    indices_to_drop = []
    for i in range(len(digit_finds)):
        digit, coords, corr = digit_finds[i]
        if abs(coords[1] - last_col) < 5:
            if i > 0 and last_corr < corr:
                indices_to_drop.append(i - 1)
            elif last_corr >= corr:
                indices_to_drop.append(i)
                continue  # keep last_corr and last_col
        last_corr = corr
        last_col = coords[1]
    for index in indices_to_drop[::-1]:
        digit_finds.pop(index)

    assert len(digit_finds) <= 2

    try:
        return int(''.join(str(find[0]) for find in digit_finds))
    except ValueError:
        return 0


def parse_puzzle(image, color=False):
    brightness = image.mean(axis=2)

    regions = get_puzzle_regions(image, color=color)
    for name, region in regions.items():
        print(name, region)

    grid = crop(brightness.astype(int), regions['grid'])

    row_lines, row_height = get_gridline_indices(grid, axis=0)
    col_lines, col_width = get_gridline_indices(grid, axis=1)
    pct_size_diff = 100 * (row_height - col_width) / col_width
    if DEBUG:
        print(f'Row height and col width differ by {pct_size_diff:.2f}%')

    unit_size = (row_height + col_width) / 2

    row_numbers_crop = crop(image, regions['row_numbers'])
    col_numbers_crop = crop(image, regions['col_numbers'])
    max_row_numbers = int(round(row_numbers_crop.shape[1] / unit_size))
    max_col_numbers = int(round(col_numbers_crop.shape[0] / unit_size))

    assert unit_size > 14  # sizes below this parse inconsistently

    row_numbers = []
    col_numbers = []
    for i, row_line in enumerate(row_lines):
        if row_line + row_height <= row_numbers_crop.shape[0]:
            row_numbers.append([])
            single_row_numbers = row_numbers_crop[row_line:row_line + int(0.95 * row_height), :]
            for j in range(max_row_numbers):
                start_col = max(single_row_numbers.shape[1] - int(round(unit_size * (j + 1))) - 1, 0)
                end_col = start_col + int(round(unit_size)) + 1
                number_crop = single_row_numbers[:, start_col:end_col]
                number = parse_number(number_crop, unit_size)
                if number != 0:
                    row_numbers[-1].append(number)
            row_numbers[-1] = row_numbers[-1][::-1]

    for i, col_line in enumerate(col_lines):
        if col_line + col_width <= col_numbers_crop.shape[1]:
            col_numbers.append([])
            single_col_numbers = col_numbers_crop[:, col_line:col_line + int(0.95 * col_width) + 1, :]
            for j in range(max_col_numbers):
                start_row = max(single_col_numbers.shape[0] - int(round(unit_size * (j + 1))) - 1, 0)
                end_row = single_col_numbers.shape[0] - int(round(unit_size * j)) + 1
                number_crop = single_col_numbers[start_row:end_row, :]
                number = parse_number(number_crop, unit_size)
                if number != 0:
                    col_numbers[-1].append(number)
            col_numbers[-1] = col_numbers[-1][::-1]

    # center of first grid square
    origin = [
        int(round(row_lines[0] + regions['grid'][0][0] + unit_size / 2)),
        int(round(col_lines[0] + regions['grid'][1][0] + unit_size / 2)),
    ]

    return row_numbers, col_numbers, origin, unit_size


def find_leftmost_group_ends(line, groups):
    i = 0
    leftmost_ends = []
    for group in groups:
        assert i + group <= len(line)
        cleared_mask = line[i:i + group] == 0
        while np.any(cleared_mask):
            i += (np.nonzero(cleared_mask)[0][0] + 1)
            assert i + group <= len(line)
            cleared_mask = line[i:i + group] == 0
        while i + group < len(line) and line[i + group] == 1:
            i += 1
        leftmost_ends.append(i + group)
        i += (group + 1)
    return leftmost_ends


def find_existing_starts_and_ends(line):
    transitions = np.diff(np.hstack([0, (line == 1).astype(np.int8), 0]))
    starts = np.nonzero(transitions == 1)[0]
    ends = np.nonzero(transitions == -1)[0]
    return starts, ends


def solve_inner_line(line, groups):
    if len(line) == 0 or len(groups) == 0:
        return line
    unsolved_indices = np.nonzero(line == -1)[0]
    if len(unsolved_indices) == 0:
        return line

    solved_start = line[:unsolved_indices[0]]
    solved_end = line[unsolved_indices[-1] + 1:]

    starts, ends = find_existing_starts_and_ends(solved_start)
    last_solved_group_in_start = -1
    starting_unsolved_index = 0
    for i in range(min(len(starts), len(groups))):
        if ends[i] - starts[i] == groups[i]:  # if False, last group in start is incomplete
            last_solved_group_in_start = i
            starting_unsolved_index = ends[i] + 1

    starts, ends = find_existing_starts_and_ends(solved_end)
    first_solved_group_in_end = len(groups)
    ending_unsolved_index = len(line)
    for i in range(min(len(starts), len(groups))):
        back_index = -i - 1
        if ends[back_index] - starts[back_index] == groups[back_index]:
            first_solved_group_in_end = len(groups) + back_index
            ending_unsolved_index = starts[back_index] + unsolved_indices[-1]

    inner_groups = groups[last_solved_group_in_start + 1:first_solved_group_in_end]
    inner_line = line[starting_unsolved_index: ending_unsolved_index]

    if starting_unsolved_index == 0 and ending_unsolved_index == len(line):
        return line

    solve_line(inner_line, inner_groups)
    return line

def assert_line_valid(line, groups):
    if np.all(line != -1):
        starts, ends = find_existing_starts_and_ends(line)
        assert len(starts) == len(groups)
        assert len(ends) == len(groups)
        for i in range(len(groups)):
            assert ends[i] - starts[i] == groups[i]

def solve_line(line, groups):
    # all necessary squares are filled
    transitions = np.diff(np.hstack([0, (line == 1).astype(np.int8), 0]))
    starts = np.nonzero(transitions == 1)[0]
    ends = np.nonzero(transitions == -1)[0]
    if len(starts) == len(groups) and np.all(ends - starts == groups):
        line[line == -1] = 0
        assert_line_valid(line, groups)
        return line

    if len(groups) == 1 and np.any(line == 1):  # single group case
        impossible_before = max(ends[0] - groups[0], 0)
        impossible_at = starts[0] + groups[0]
        line[:impossible_before] = 0
        line[impossible_at:] = 0

    overlap_ends = find_leftmost_group_ends(line, groups)
    overlap_starts = [
        len(line) - back_index
        for back_index in find_leftmost_group_ends(line[::-1], groups[::-1])
    ][::-1]

    # entire ranges where filled squares in a group could be
    possible_starts = [end - group for end, group in zip(overlap_ends, groups)]
    possible_ends = [start + group for start, group in zip(overlap_starts, groups)]

    if len(groups) > 0:
        filled_squares_near_start = np.nonzero(line[possible_starts[0]:possible_ends[0]] == 1)[0]
        if len(filled_squares_near_start) > 0:
            first_filled_square = filled_squares_near_start[0] + possible_starts[0]
            possible_ends[0] = min(possible_ends[0], first_filled_square + groups[0])
            overlap_starts[0] = min(overlap_starts[0], first_filled_square)

        filled_squares_near_end = np.nonzero(line[possible_starts[-1]:possible_ends[-1]] == 1)[0]
        if len(filled_squares_near_end) > 0:
            last_filled_square = filled_squares_near_end[-1] + possible_starts[-1]
            possible_starts[-1] = max(possible_starts[-1], last_filled_square - groups[-1] + 1)
            overlap_ends[-1] = max(overlap_ends[-1], last_filled_square + 1)

    #print(np.vstack([overlap_starts, overlap_ends]).T)
    #print(np.vstack([possible_starts, possible_ends]).T)

    # apply overlap method
    if len(groups) > 0:
        line[:max(overlap_ends[0] - groups[0], 0)] = 0
        line[overlap_starts[-1] + groups[-1]:] = 0
        for overlap_start, overlap_end in zip(overlap_starts, overlap_ends):
            line[overlap_start:overlap_end] = 1

    for start, end in zip(starts, ends):
        length = end - start
        larger_group_possible = False
        for possible_start, possible_end, group in zip(possible_starts, possible_ends, groups):
            if end > possible_end:
                continue
            if start < possible_start:
                break
            if group > length:
                larger_group_possible = True
                break
        if not larger_group_possible:
            if start > 0:
                line[start - 1] = 0
            if end < len(line):
                line[end] = 0

    # index of black squares with white squares directly left
    white_to_black_indices = np.nonzero((line[:-1] == 0) & (line[1:] == 1))[0] + 1
    for index in white_to_black_indices:
        min_possible_group = np.inf
        for possible_start, possible_end, group in zip(possible_starts, possible_ends, groups):
            if index >= possible_end:
                continue
            if possible_start > index:
                break
            if group < min_possible_group:
                min_possible_group = group
        assert min_possible_group != np.inf
        line[index:index + min_possible_group] = 1

    # index of black squares with white squares directly right
    black_to_white_indices = np.nonzero((line[:-1] == 1) & (line[1:] == 0))[0]
    for index in black_to_white_indices:
        min_possible_group = np.inf
        for possible_start, possible_end, group in zip(possible_starts, possible_ends, groups):
            if index >= possible_end:
                continue
            if possible_start > index:
                break
            if group < min_possible_group:
                min_possible_group = group
        assert min_possible_group != np.inf
        line[index - min_possible_group + 1:index] = 1

    solve_inner_line(line, groups)

    assert_line_valid(line, groups)
    return line

def solve_puzzle(puzzle_state, row_numbers, col_numbers, start_time=None, depth=0):
    global MAX_DEPTH
    if depth > MAX_DEPTH:
        MAX_DEPTH = depth
        if depth > 100:
            print(f'recursion depth: {depth}')

    if start_time is None:
        start_time = time.time()

    timeout = 5 if INTERACTIVE else 120
    if time.time() - start_time > timeout:
        return puzzle_state

    while True:
        last_puzzle_state = puzzle_state.copy()
        for j in range(puzzle_state.shape[0]):
            solve_line(puzzle_state[j, :], row_numbers[j])
            solve_line(puzzle_state[j, :], row_numbers[j])
        if np.all(puzzle_state != -1):
            break

        for j in range(puzzle_state.shape[1]):
            solve_line(puzzle_state[:, j], col_numbers[j])
            solve_line(puzzle_state[:, j], col_numbers[j])
        if np.all(puzzle_state != -1):
            break
        if np.all(puzzle_state == last_puzzle_state):
            break

    if not np.all(puzzle_state != -1):
        first_unsolved = np.nonzero(puzzle_state.flatten() == -1)[0][0]
        first_unsolved_rc = np.unravel_index(first_unsolved, puzzle_state.shape)
        puzzle_state[first_unsolved_rc] = 0
        puzzle_state_copy = puzzle_state.copy()
        try:
            puzzle_state = solve_puzzle(puzzle_state, row_numbers, col_numbers, start_time=start_time, depth=depth+1)
        except AssertionError:
            puzzle_state = puzzle_state_copy
            puzzle_state[first_unsolved_rc] = 1
            puzzle_state = solve_puzzle(puzzle_state, row_numbers, col_numbers, start_time=start_time, depth=depth+1)
        if not np.all(puzzle_state != -1):  # sub-call timed out
            puzzle_state = puzzle_state_copy

    for i in range(puzzle_state.shape[0]):
        assert_line_valid(puzzle_state[i, :], row_numbers[i])
    for i in range(puzzle_state.shape[1]):
        assert_line_valid(puzzle_state[:, i], col_numbers[i])

    return puzzle_state


if __name__ == '__main__':
    path = os.path.join(SCRIPT_DIR, "screenshot.png")

    image = imread(path)[:, :, :3]  # strip alpha channel

    try:
        row_numbers, col_numbers, origin, unit_size = parse_puzzle(image)
    except AssertionError:
        print('Could not solve puzzle.')
        raise

    if DEBUG:
        print(f'Row numbers ({len(row_numbers)}):')
        for line in row_numbers:
            print(','.join(str(x) for x in line))

    if DEBUG:
        print(f'Col numbers ({len(col_numbers)}):')
        for line in col_numbers:
            print(','.join(str(x) for x in line))

    puzzle_state = np.full((len(row_numbers), len(col_numbers)), -1)
    puzzle_state = solve_puzzle(puzzle_state, row_numbers, col_numbers)
    fully_solved = np.all(puzzle_state != -1)

    print(puzzle_state_str(puzzle_state))

    while not fully_solved and INTERACTIVE:
        row = int(input(f'Row (1 - {puzzle_state.shape[0]}) to set value at: ')) - 1
        col = int(input(f'Column (1 - {puzzle_state.shape[0]}) to set value at: ')) - 1
        value = int(input(f'Value (1 - {puzzle_state.shape[0]}) to set: '))
        puzzle_state[row, col] = value

        puzzle_state = solve_puzzle(puzzle_state, row_numbers, col_numbers)

        fully_solved = np.all(puzzle_state != -1)
        print(puzzle_state_str(puzzle_state))

    if not fully_solved:
        print('Partially solved.')

    if fully_solved or FORCE_SAVE:
        touch_locations = (np.argwhere(puzzle_state == 1) * unit_size + origin).astype(int)

        with open(os.path.join(SCRIPT_DIR, "touch_locations.txt"), 'w') as f:
            f.write(
                '\n'.join(
                    ' '.join(str(coord) for coord in touch_locations[i, ::-1])  # switch from rc to xy
                    for i in range(touch_locations.shape[0])
                )
            )
