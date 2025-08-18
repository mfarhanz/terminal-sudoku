from random import random, choice, randint
from time import sleep, time
from threading import Timer, Thread
from signal import SIGINT
from pyfiglet import Figlet
from itertools import product
import keyboard
import platform
import os
import sys

def draw_grid(matrix=None, no_delay=False):
    rows = len(matrix)
    cols = len(matrix[0])
    delay = 0 if no_delay else 0.0001
    sys.stdout.write(f'\n{VDASH_T}')
    for c in range(cols):
        sys.stdout.write(HDASH * 5 + (VDASH_T, VDASH_Tv)[c in [2, 5]])
        sys.stdout.flush()
        sleep(delay)
    print()
    for row in range(rows - 1):
        sys.stdout.write(VDASH_M)
        for c, x in enumerate(matrix[row]):
            sys.stdout.write(f"{('  ' + PRESET + (str(x) if x != 0 else ' ') + RESET + '  ' + (VDASH_M, VDASH_Mv)[c in [2, 5]])}")
            sys.stdout.flush()
            sleep(delay)
        sys.stdout.write(f"\n{VDASH_M}")
        for c in range(cols):
            sys.stdout.write(f"{((HDASH, HDASH_Hv)[row in [2, 5]]) * 5 + (VDASH_M, VDASH_Mv)[c in [2, 5]]}")
            sys.stdout.flush()
            sleep(delay)
        print()
    sys.stdout.write(VDASH_M)
    for c, x in enumerate(matrix[-1]):
        sys.stdout.write(f"{('  ' + PRESET + (str(x) if x != 0 else ' ') + RESET + '  ' + (VDASH_M, VDASH_Mv)[c in [2, 5]])}")
        sys.stdout.flush()
        sleep(delay)
    sys.stdout.write(f'\n{VDASH_B}')
    for c in range(cols):
        sys.stdout.write(HDASH * 5 + (VDASH_B, VDASH_Bv)[c in [2, 5]])
        sys.stdout.flush()
        sleep(delay)
    # sys.stdout.flush()
    print()

def update_cell(row, col, val, no_delay=False, delay=0.2):
    def overwrite():
        # Move to the correct position using ANSI escape codes
        sys.stdout.write(f"\033[{(row * 2) + 3};{(col * 6) + 3}H")  # Row and column positions are adjusted
        # Backspace the current value (overwrite with space)
        sys.stdout.write("\033[1D")  # Move left by 1
        sys.stdout.write(" ")  # Overwrite with a space
        sys.stdout.write("\033[1D")  # Move left by 1 to the position of the number
        # Write the new number in the same position
        sys.stdout.write(f"  {val if int(val) != 0 else ' '}  ")
        update_cell_border(row, col)
    
    if (row, col) in CLUE_LOCS:
        flash_cell(row, col, RED, 0.1, 4)
    else:
        sys.stdout.write(GREEN)  # Highlight cell
        overwrite()
        if not no_delay:
            sleep(delay)  # Brief pause for the color effect
        sys.stdout.write(RESET)  # Reset to normal colors after the effect
        overwrite()
        GRID[row][col] = int(val)
        if check_sudoku():
            display_temp_message("puzzle solved", delay=2)

def flash_cell(row, col, color, period=1.0, repeat=1):
    def flash(count):
        if count >= repeat:
            return  # Stop if we've reached the repeat count
            # Alternate flashing on and off
        highlight_cell(row, col, color=color)
        # Schedule the next flash off after the period
        Timer(period, highlight_cell, [row, col, None]).start()
        # Schedule the next flash after the period (the next color on)
        Timer(period * 2, flash, [count + 1]).start()
    
    global DELAY1_TIMER
    if DELAY1_TIMER is not None:
        DELAY1_TIMER.cancel()
    DELAY1_TIMER = Timer(0, flash, [0])
    DELAY1_TIMER.start()

def highlight_cell(row, col, highlight=True, color=None):
    if color:
        sys.stdout.write(color if highlight else RESET)  # Set color
    else:
        sys.stdout.write(BLUE if highlight else RESET)
    # Move to the correct position using ANSI escape codes
    sys.stdout.write(f"\033[{(row * 2) + 3};{(col * 6) + 3}H")  # Row and column positions are adjusted
    update_cell_border(row, col)

def update_cell_border(row, col):
    div_upper_l = (VDASH_T, VDASH_Tv)[col in [3, 6]] if row == 0 else (VDASH_M, VDASH_Mv)[col in [3, 6]]
    div_upper_r = (VDASH_T, VDASH_Tv)[col in [2, 5]] if row == 0 else (VDASH_M, VDASH_Mv)[col in [2, 5]]
    div_lower_l = (VDASH_B, VDASH_Bv)[col in [3, 6]] if row == 8 else (VDASH_M, VDASH_Mv)[col in [3, 6]]
    div_lower_r = (VDASH_B, VDASH_Bv)[col in [2, 5]] if row == 8 else (VDASH_M, VDASH_Mv)[col in [2, 5]]
    # Rewrite the surrounding borders (dashes and dividers)
    sys.stdout.write(f"\033[{(row * 2) + 2};{(col * 6) + 1}H")  # Move to the top border
    sys.stdout.write(f"{div_upper_l + (HDASH, HDASH_Hv)[row in [3, 6]] * 5 + div_upper_r}")  # Top border with new color
    sys.stdout.write(f"\033[{(row * 2) + 4};{(col * 6) + 1}H")  # Move to the bottom border
    sys.stdout.write(
        f"{div_lower_l + (HDASH, HDASH_Hv)[row in [2, 5]] * 5 + div_lower_r}")  # Bottom border with new color
    # Move to the left divider position (start of the cell)
    sys.stdout.write(f"\033[{(row * 2) + 3};{(col * 6) + 1}H")  # Left divider (before the number)
    sys.stdout.write((VDASH_M, VDASH_Mv)[col in [3, 6]])  # Left divider with new color
    sys.stdout.write(f"\033[{(row * 2) + 3};{(col * 6) + 7}H")  # Move to the right border (divider)
    sys.stdout.write((VDASH_M, VDASH_Mv)[col in [2, 5]])  # Right divider with new color
    sys.stdout.flush()

########### CODE COPIED FROM OUTSIDE - BEGIN ###########
"""
this section of code has been copied from https://www.cs.mcgill.ca/~aassaf9/python/sudoku.txt
"""

def knuth_algorithm_x_solver(visuals=True):
    global GRID, AUTO_SOLVE
    if check_sudoku():
        return          # if grid is already solved, return
    R, C = 3, 3
    N = R * C
    X = ([("rc", rc) for rc in product(range(N), range(N))] +
         [("rn", rn) for rn in product(range(N), range(1, N + 1))] +
         [("cn", cn) for cn in product(range(N), range(1, N + 1))] +
         [("bn", bn) for bn in product(range(N), range(1, N + 1))])
    Y = dict()
    try:
        for r, c, n in product(range(N), range(N), range(1, N + 1)):
            b = (r // R) * R + (c // C)  # Box number
            Y[(r, c, n)] = [
                ("rc", (r, c)),
                ("rn", (r, n)),
                ("cn", (c, n)),
                ("bn", (b, n))]
        X, Y = exact_cover(X, Y)
        for i, row in enumerate(GRID):
            for j, n in enumerate(row):
                if n:
                    select(X, Y, (i, j, n))
        for solution in x_solve(X, Y, []):
            for (r, c, n) in solution:
                if not AUTO_SOLVE:
                    return
                if visuals:
                    update_cell(r, c, n)
                else:
                    GRID[r][c] = n
            AUTO_SOLVE = False
            return
    except (KeyError, IndexError, TypeError, ValueError) as e:
        AUTO_SOLVE = False
        display_temp_message(choice(ERROR_MSGS))

def exact_cover(X, Y):
    X = {j: set() for j in X}
    for i, row in Y.items():
        for j in row:
            X[j].add(i)
    return X, Y

def x_solve(X, Y, solution):
    if not X:
        yield list(solution)
    else:
        c = min(X, key=lambda c: len(X[c]))
        for r in list(X[c]):
            solution.append(r)
            cols = select(X, Y, r)
            for s in x_solve(X, Y, solution):
                yield s
            deselect(X, Y, r, cols)
            solution.pop()

def select(X, Y, r):
    cols = []
    for j in Y[r]:
        for i in X[j]:
            for k in Y[i]:
                if k != j:
                    X[k].remove(i)
        cols.append(X.pop(j))
    return cols

def deselect(X, Y, r, cols):
    for j in reversed(Y[r]):
        X[j] = cols.pop()
        for i in X[j]:
            for k in Y[i]:
                if k != j:
                    X[k].add(i)

def brute_force_solver():
    def is_valid(g, r, c, n):
        # Check row
        if n in g[r]:
            return False
        # Check column
        if n in [g[i][c] for i in range(9)]:
            return False
        # Check 3x3 box
        start_row, start_col = 3 * (r // 3), 3 * (c // 3)
        for r in range(start_row, start_row + 3):
            for c in range(start_col, start_col + 3):
                if g[r][c] == n:
                    return False
        return True

    def find_empty_cell(g):
        for r in range(9):
            for c in range(9):
                if g[r][c] == 0:
                    return r, c
        return None
    
    global GRID, AUTO_SOLVE
    if check_sudoku():
        AUTO_SOLVE = False
        return False
    empty = find_empty_cell(GRID)
    if not empty:
        AUTO_SOLVE = False
        return True  # Solved
    row, col = empty
    for num in range(1, 10):
        if not AUTO_SOLVE:
            return False
        if is_valid(GRID, row, col, num):
            update_cell(row, col, num, delay=0.07)
            if brute_force_solver():
                return True
            if not AUTO_SOLVE:
                return False
            update_cell(row, col, 0, delay=0.07)
    return False

########### CODE COPIED FROM OUTSIDE - END ###########

def check_digit_list(lst, strict=True):
    if strict:
        return set(lst) == set(range(1, 10))
    else:
        seen = set()
        for digit in lst:
            if digit != 0:
                if digit in seen:
                    return False
                seen.add(digit)
        return True

def check_sudoku(partial=False):
    for row in GRID:
        if not check_digit_list(row, not partial):
            return False
    for col in zip(*GRID):
        if not check_digit_list(col, not partial):
            return False
    for cy in [0, 3, 6]:
        for cx in [0, 3, 6]:
            box = [GRID[y][x] for y in range(cy, cy + 3) for x in range(cx, cx + 3)]
            if not check_digit_list(box, not partial):
                return False
    return True

def check_validity_at_row_col_box(r, c):
    bcx = (r // 3) * 3 + 1
    bcy = (c // 3) * 3 + 1
    row_c = check_digit_list(GRID[r], strict=False)
    col_c = check_digit_list([GRID[i][c] for i in range(9)], strict=False)
    box_c = check_digit_list([GRID[x][y] for x in range(bcx - 1, bcx + 2) for y in range(bcy - 1, bcy + 2)],
                             strict=False)
    return (row_c << 2) | (col_c << 1) | box_c

def check_validity_of_value(r, c):
    val = GRID[r][c]
    bcx = (r // 3) * 3 + 1
    bcy = (c // 3) * 3 + 1
    index = {i: 0 for i in range(1, 10)}
    for x in GRID[r]:
        if x:
            index[x] += 1
    for x in [GRID[i][c] for i in range(9) if i != r]:
        if x:
            index[x] += 1
    for x in [GRID[x][y] for x in range(bcx - 1, bcx + 2) for y in range(bcy - 1, bcy + 2) if x != r and y != c]:
        if x:
            index[x] += 1
    return True if index[val] == 1 else False

def get_random_correct_entry():
    global AUTO_SOLVE
    if check_sudoku(partial=True):
        AUTO_SOLVE = True   # need to do this unfortunately due to bad code design
        original = [row[:] for row in GRID]
        knuth_algorithm_x_solver(visuals=False)
        x, y = randint(0, 8), randint(0, 8)
        val = GRID[x][y]
        for j in range(9):
            for i in range(9):
                GRID[j][i] = original[j][i]
        return x, y, val
    else:
        return
    
def update_message(msg, mx=22, my=2, delay=None):
    global MESSAGE
    if MESSAGE:
        if delay is None:
            show_message(MESSAGE, sr=mx, sc=my, color=BLACK)
        else:
            show_message(MESSAGE, sr=mx, sc=my, color=BLACK, delay=delay)
    MESSAGE = msg

def show_message(msg, delay=0.001, sr=None, sc=None, color=None, no_delay=False):
    start_row, start_col = 22 if not sr else sr, 2 if not sc else sc
    char_lines = []
    current_font = None
    for c in msg:
        target_font = NUM_FONT if c.isdigit() else DEF_FONT if c.isalpha() else SYM_FONT
        if current_font != target_font:
            RENDERER.setFont(font=target_font)
            current_font = target_font
        char_lines.append(RENDERER.renderText(c).splitlines())
    # Estimate the height of a character block
    height = len(char_lines[0])
    # Initialize empty output lines
    output_lines = ['' for _ in range(height)]
    if color is not None:
        sys.stdout.write(color)
    chunk_size = 24
    chunks = [char_lines[i:i + chunk_size] for i in range(0, len(char_lines), chunk_size)]
    for block in chunks:
        for char in block:
            for i in range(height):
                output_lines[i] += char[i]
            for i, line in enumerate(output_lines):
                sys.stdout.write(f"\033[{start_row + i};{start_col}H{line}")
            sys.stdout.flush()
            if not no_delay:
                sleep(delay)
        output_lines[:] = ['' for _ in range(height)]
        start_row += 3
    if color is not None:
        sys.stdout.write(RESET)

def input_mode_blink():
    global DELAY2_TIMER, SHOW_CURSOR_ON_BLINK

    if not INPUT_MODE:
        # Clean up and optionally clear message
        if DELAY2_TIMER:
            DELAY2_TIMER.cancel()
            DELAY2_TIMER = None
        # show_message(INPUT + '   ', color=CONSOLE, no_delay=True)
        return
    # Alternate between INPUT and INPUT + '_'
    display = INPUT + ('_' if SHOW_CURSOR_ON_BLINK else '   ')
    show_message(display, color=CONSOLE, no_delay=True)
    SHOW_CURSOR_ON_BLINK = not SHOW_CURSOR_ON_BLINK
    # Schedule next toggle
    DELAY2_TIMER = Timer(0.6, input_mode_blink)
    DELAY2_TIMER.start()

def display_temp_message(msg, delay=None, c_delay=None):
    update_message(msg)
    show_message(MESSAGE, delay=c_delay if c_delay else 0.001)
    sleep(delay if delay else 1)
    update_message('', delay=c_delay)

def display_splash():
    update_message("TERMINAL SUDOKU")
    show_message(MESSAGE, sr=10, sc=5)
    sleep(2)
    update_message('', 10, 5)
    clear_screen()

def clear_screen():
    # Check the operating system
    if platform.system().lower() == "windows":
        os.system("cls")  # For Windows
    else:
        os.system("clear")  # For Unix-like systems (Linux/macOS)
    sys.stdout.write("\033[?25l")
    sys.stdout.flush()

def reset_console():
    sys.stdout.write(f"\033[{(13 * 2) + 3};{(0 * 6) + 3}H")
    sys.stdout.write("\033[?25h")
    sys.stdout.write(RESET)
    sys.stdout.flush()

# Background thread function
def inactivity_checker(timeout_seconds=300):
    global TIMESTAMP
    while True:
        sleep(10)  # Check every 5 seconds — low CPU usage
        elapsed = time() - TIMESTAMP
        if elapsed >= timeout_seconds:
            display_temp_message(choice(AFK_MSGS), 2)
            # Reset so the message doesn't spam
            TIMESTAMP = time()

def end_process():
    global AUTO_SOLVE, AUTO_SOLVER_CHOOSING, INPUT_MODE, DELAY1_TIMER, DELAY2_TIMER
    AUTO_SOLVE = False
    AUTO_SOLVER_CHOOSING = False
    INPUT_MODE = False
    DELAY1_TIMER = None
    DELAY2_TIMER = None
    sys.stdout.write(RESET)
    display_temp_message(choice(QUIT_MSGS), 1.7, 0.08)
    reset_console()
    keyboard.unhook_all()  # Stop the event listener
    os.kill(os.getpid(), SIGINT)

def on_key_event(event):
    global CURR_Y, CURR_X, GRID, CLUE_LOCS, AUTO_SOLVE, AUTO_SOLVER_CHOOSING, INPUT_MODE, INPUT, TIMESTAMP
    TIMESTAMP = time()
    if event.name in ['up', 'down', 'right', 'left'] or event.name in '123456789':
        if not AUTO_SOLVE:
            highlight_cell(CURR_Y, CURR_X, False)
            if event.name == 'up':
                CURR_Y = (CURR_Y - 1) % 9
            elif event.name == 'down':
                CURR_Y = (CURR_Y + 1) % 9
            elif event.name == 'left':
                CURR_X = (CURR_X - 1) % 9
            elif event.name == 'right':
                CURR_X = (CURR_X + 1) % 9
            elif event.name in '123456789':
                update_cell(CURR_Y, CURR_X, event.name)
            highlight_cell(CURR_Y, CURR_X, True)
    elif event.name in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ':
        if INPUT_MODE and len(INPUT) < 20:
            sys.stdout.write(CONSOLE)
            INPUT += event.name
            show_message(INPUT, no_delay=True)
            sys.stdout.write(RESET)
    elif event.name == 'backspace':
        if INPUT_MODE:
            INPUT = INPUT[:-1]
            show_message(INPUT + '     ', no_delay=True)
            sys.stdout.write(RESET)
        else:
            if not AUTO_SOLVE:
                update_cell(CURR_Y, CURR_X, 0)
    elif event.name == 'enter':
        sys.stdout.write(RESET)
        if not INPUT_MODE:
            INPUT_MODE = True
            input_mode_blink()
        else:
            INPUT_MODE = False
            if any(sub in INPUT for sub in ['restart', 'res', 'board', 're', 'new', 'gen', 'create', 'blan']):
                if not AUTO_SOLVE:
                    GRID[:], CLUE_LOCS[:] = generate_sudoku()
                    clear_screen()
                    draw_grid(GRID)
                    update_message('new grid generated')
                    show_message(MESSAGE)
                    sleep(1)
                    update_message('')
            elif any(sub in INPUT for sub in ['cls']):
                clear_screen()
                draw_grid(GRID, no_delay=True)
            elif any(sub in INPUT for sub in ['rul', 'goa', 'direct', 'aim']):
                info = "each row, column, and box must contain the digits 1 to 9 exactly once."
                display_temp_message(info, delay=3)
            elif any(sub in INPUT for sub in ['help', 'hlp', 'guid', '?', 'inf']):
                info = "use arrow keys to move. use num keys to solve. hit enter to type commands. type solve for auto-solver. hit esc to quit."
                display_temp_message(info, delay=4)
            elif any(sub in INPUT for sub in ['comman', 'control', 'key', 'usag']):
                info = "[help/guide/info/?][rules/goal][commands/usage][cls/clear][reset/new/blank][esc/quit/exit][hint/tip/advice][solve/auto][stop/break/cancel/x][slow/fast]"
                display_temp_message(info, delay=10)
            elif any(sub in INPUT for sub in ['stop', 'paus', 'break', 'halt', 'wait', 'x']):
                if AUTO_SOLVE:
                    AUTO_SOLVE = False
                    AUTO_SOLVER_CHOOSING = False
                    display_temp_message("auto-solver stopped.")
            elif any(sub in INPUT for sub in ['sol', 'auto', 'ai', 'start', 'ans', 'commen',
                                              'begin', 'solv', 'sol', 'aut', 'find', 'fini']):
                if not AUTO_SOLVE:
                    update_message('auto-solver chosen')
                    show_message(MESSAGE)
                    AUTO_SOLVE = True
                    sleep(1)
                    update_message('choose method - slow or fast?')
                    show_message(MESSAGE)
                    AUTO_SOLVER_CHOOSING = True
            elif any(sub in INPUT for sub in ['slo', 'fas', 'low', 'spee', 'rapi', 'quic', 'zip', 'hurr']):
                if AUTO_SOLVE and AUTO_SOLVER_CHOOSING:
                    AUTO_SOLVER_CHOOSING = False
                    update_message('enter x to stop.')
                    show_message(MESSAGE)
                    if any(sub in INPUT for sub in ['low', 'slo', 'easy', 'stead', 'ez']):
                        th = Thread(target=brute_force_solver, daemon=True)
                        th.start()
                    else:
                        th = Thread(target=knuth_algorithm_x_solver, daemon=True)
                        th.start()
                else:
                    display_temp_message("unexpected command.")
            elif any(sub in INPUT for sub in ['cross', 'canc']):
                if AUTO_SOLVE and AUTO_SOLVER_CHOOSING:
                    AUTO_SOLVE = False
                    AUTO_SOLVER_CHOOSING = False
                    display_temp_message("auto-solver stopped.")
                else:
                    display_temp_message("no task found to cancel")
            elif any(sub in INPUT for sub in ['quit', 'exi', 'end', 'esc', 'leav', 'termi', 'kill', 'clos',
                                              'abor']):
                end_process()
            elif any(sub in INPUT for sub in ['hint', 'assist', 'clu', 'point', 'tip', 'cue', 'sugg', 'how',
                                              'advi', 'recomm', 'sos']):
                p = random()
                try:
                    if 0 <= p <= 0.05:
                        entry = get_random_correct_entry()
                        if entry is None:
                            update_message("damn. somethings wrong. gl.")
                        else:
                            update_message(f"set [{entry[0]}][{entry[1]}] = {entry[2]} . trust.")
                    elif 0.06 <= p <= 0.25:
                        valid = check_sudoku(partial=True)
                        if valid:
                            update_message("hint: the grid is still good rn")
                        else:
                            update_message("hint: grid needs fixing")
                    elif 0.26 <= p <= 0.5:
                        valid = check_validity_of_value(CURR_Y, CURR_X)
                        if valid:
                            update_message(f"value at [{CURR_Y}][{CURR_X}] is valid")
                        else:
                            update_message(f"value at [{CURR_Y}][{CURR_X}] repeats")
                    elif 0.51 <= p <= 0.75:
                        labels = ['row', 'col', 'box']
                        valid = check_validity_at_row_col_box(CURR_Y, CURR_X)
                        res = [(valid >> i) & 1 for i in (2, 1, 0)]
                        valid_parts = [label for label, valid in zip(labels, res) if valid]
                        invalid_parts = [label for label, valid in zip(labels, res) if not valid]
                        if len(valid_parts) == 3:
                            update_message("row/col/box valid here")
                        elif len(valid_parts) == 0:
                            update_message("row/col/box all invalid here")
                        elif len(valid_parts) == 1:
                            update_message(f"{valid_parts[0]} valid, but {'/'.join(invalid_parts)} not valid")
                        elif len(valid_parts) == 2:
                            update_message(f"{'/'.join(valid_parts)} valid, but {invalid_parts[0]} not valid")
                        else:
                            update_message("need to check row/col/box here")
                    elif 0.76 <= p <= 1.0:
                        valid = check_sudoku()
                        if valid:
                            update_message("grid's already solved, dawg.")
                        else:
                            update_message("grid ain't solved. hope that helps.")
                except Exception as e:
                    update_message(choice(ERROR_MSGS))
                show_message(MESSAGE)
                sleep(2)
                update_message('')
            else:
                display_temp_message(choice(INPUT_ERROR_MSGS))
            INPUT = ''
    elif event.name == 'esc':  # Clean exit if ESC is pressed
        end_process()

def generate_sudoku():
    def available_digits(r, c):
        available = [n for n in range(1, 10)]
        for x in g[r]:
            if x in available:
                available.remove(x)
        for y in [g[x][c] for x in range(9)]:
            if y in available:
                available.remove(y)
        bcx = (r // 3) * 3 + 1
        bcy = (c // 3) * 3 + 1
        box = [g[x][y] for x in range(bcx - 1, bcx + 2) for y in range(bcy - 1, bcy + 2)]
        for n in box:
            if n in available:
                available.remove(n)
        return available
    
    g = [[0 for _ in range(9)] for _ in range(9)]
    f = []
    for i in range(9):
        for j in range(9):
            if random() < 0.1:
                g[i][j] = choice(available_digits(i, j))
                f.append((i, j))
    return g, f

if __name__ == '__main__':
    GREEN = '\033[32m'
    RED = '\033[31m'
    BLUE = '\033[34m'
    BLACK = '\033[30m'
    CONSOLE = '\033[38;2;13;188;121m'
    PRESET = '\033[38;2;0;191;255m'
    RESET = '\033[0m'
    HDASH = '\u2014'
    HDASH_Hv = '\u2501'
    VDASH_T = '\u2577'
    VDASH_Tv = '\u257B'
    VDASH_M = '\u2502'
    VDASH_Mv = '\u2503'
    VDASH_B = '\u2575'
    VDASH_Bv = '\u2579'
    GRID = None
    CLUE_LOCS = None
    DELAY1_TIMER = None
    DELAY2_TIMER = None
    TIMESTAMP = None
    SHOW_CURSOR_ON_BLINK = False
    RENDERER = None
    DEF_FONT = 'cybermedium'
    NUM_FONT = 'straight'   # or 'mini'
    SYM_FONT = 'italic'     # or 'mini'
    AUTO_SOLVE = False
    AUTO_SOLVER_CHOOSING = False
    AUTO_SOLVER_METHOD = 'a'
    INPUT_MODE = False
    INPUT = ''
    MESSAGE = ''
    CURR_X = 0
    CURR_Y = 0
    
    QUIT_MSGS = [
        'bye bye...', 'exiting...', 'good bye', 'zzz...', 'shutting down', 'system failure', 'aborting...',
        'process terminated', 'goodbye', 'connection lost', 'trace complete. exiting.', 'rage quit',
        'gg wp', 'bye', 'farewell', 'ciao', 'au revoir', 'until next time', 'cya', 'take care', 'we will meet again',
        'better luck next time', 'killing process', 'terminating task', 'cu next time'
    ]
    
    INPUT_ERROR_MSGS = [
        'unknown command', 'invalid command', 'command not found', 'error - unrecognized input.', 'invalid input',
        "that doesnt seem right.", 'unrecognized operation.', 'oops. try something else.', 'unknown directive',
        'nice try. i still dont get you.', 'what?', 'huh?', 'uhhh....?', 'um....?', 'not sure what you mean.',
        'pardon?', 'could you repeat that?', 'well, that was... something.',
        'what does that even mean??', "can't help you there.",
        "can't help you with that request.", "i don't know what you mean.", 'excuse me?', 'english, please.', '??????',
        "dude. what?"
    ]
    
    ERROR_MSGS = [
        'yikes. something went wrong.', 'cant be done rn. apologies.', 'endpoint crashed. sorry!',
        'uh oh. somethings wrong.', 'i crashed. somethings up.', 'umm. maybe remove dupe entries.',
        'somethings funky with this grid.', 'sorry, i crashed.', 'ran into an error. sorry.',
        "error. don't worry, its not you.", 'f**k. this shouldnt happen.', 'goddamit. function crashed.',
        "oh FFS. can't help, sorry. ", "can't continue with this request", 'shiiit. maybe see for errors?',
        "well i tried. i give up.", "aaand i crashed. back to you.", "crashed. this is embarrassing.",
        "somethings funky. i cant continue.", "uhhh... i am stuck.", "dang. looks like you're on your own.",
        "i lost the pointer. oops.", "wtf did you do??", "rip. i cant find this bug.", "i have run into a problem.",
        "bug encountered. giving up.", "crash. nooo this shouldnt have happened.", "so sorry, but i cant help rn.",
        "somethings wrong.", "something went wrong", "dang it, theres a problem.", "some shit isnt working.",
        "nevermind, something went wrong.", "nvm, the function crashed.", "damn it. shit hit the fan.",
        "damn. somethings wrong. gl.", "can't help you rn gng"
    ]
    
    AFK_MSGS = [
        'hello? anyone there?', 'gng where you at', '*crickets*', 'hmmmmmmmmmm.', 'bruhhhhhh.', 'boooooooring.',
        'yo? anyone there?', 'hellllooooo?', 'welp.', 'hello? i’m still running here.', 'still here. unlike you.',
        'classic. start me, then disappear.', 'this silence is getting weird.', 'you good? just checking.',
        'please come back.', 'at least shut me down.', "can't even ctrl+c myself.", "still waiting.",
        "guess i'll keep waiting.", "just me and my thoughts i guess.", "well, this is awkward.",
        "cool. i’ll just... wait.", "i feel... abandoned.", "guess I’ll talk to myself.", "your chair misses you."
    ]
    
    try:
        TIMESTAMP = time()
        RENDERER = Figlet(font=DEF_FONT)
        GRID, CLUE_LOCS = generate_sudoku()
        clear_screen()
        display_splash()
        draw_grid(GRID)
        sleep(1)
        bg_thread = Thread(target=inactivity_checker, daemon=True)
        bg_thread.start()
        keyboard.on_press(on_key_event, suppress=True)
        keyboard.wait('`')
    except KeyboardInterrupt:
        reset_console()
