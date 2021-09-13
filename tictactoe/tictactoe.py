import math
import copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    if board == initial_state():
        return X

    numx = 0
    numo = 0
    for row in board:
        numx += row.count(X)
        numo += row.count(O)

    if numx == numo:
        return X
    else:
        return O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    action = set()
    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                action.add((i, j))
    return action


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """

    i, j = action
    if i < 3 and j < 3 and board[i][j] is EMPTY:
        currentP = player(board)
        board2 = copy.deepcopy(board)
        board2[i][j] = currentP
        return board2
    else:
        raise Exception("Invalid action!")


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """

    # horizontal win
    for row in board:
        if row[0] != EMPTY and row.count(row[0]) == len(row):
            return row[0]

    # vertical win
    for i in range(len(board)):
        vertical = []

        for row in board:
            vertical.append(row[i])

        if vertical[0] != EMPTY and vertical.count(vertical[0]) == len(vertical):
            return vertical[0]

    # diagonal win
    diag = []
    for i in range(len(board)):
        diag.append(board[i][i])
    if diag[0] != EMPTY and diag.count(diag[0]) == len(diag):
        return diag[0]

    diag = []
    diag.append(board[0][2])
    diag.append(board[1][1])
    diag.append(board[2][0])
    if diag[0] != EMPTY and diag.count(diag[0]) == len(diag):
        return diag[0]

    return None


def terminal(board):

    if winner(board) == X or winner(board) == O:
        return True
    if any(None in row for row in board):
        return False
    else:
        return True


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if winner(board) == "X":
        return 1
    elif winner(board) == "O":
        return -1
    else:
        return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None

    myactions = actions(board)

    if player(board) == X:
        st = -math.inf
        move = (-1, -1)
        for action in myactions:
            v = minimum(result(board, action))
            if v == 1:
                move = action
                break
            if v > st:
                st = v
                move = action
        return move
    if player(board) == O:
        st = math.inf
        move = (-1, -1)
        for action in myactions:
            v = maximum(result(board, action))
            if v == -1:
                move = action
                break
            if v < st:
                st = v
                move = action
        return move


def maximum(board):
    if terminal(board):
        return utility(board)

    i = -math.inf
    newactions = actions(board)

    for action in newactions:
        oppo = minimum(result(board, action))
        i = max(i, oppo)
        if i == 1:
            break

    return i


def minimum(board):
    if terminal(board):
        return utility(board)

    i = math.inf
    newactions = actions(board)

    for action in newactions:
        oppo = maximum(result(board, action))
        i = min(i, oppo)
        if i == -1:
            break

    return i