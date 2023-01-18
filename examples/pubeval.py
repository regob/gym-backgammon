"""Interface to pubeval.c

Modified, and reimplemented from: https://github.com/weekend37/Backgammon.
"""

import numpy as np
import ctypes
from ctypes import cdll, c_float, c_int

lib = cdll.LoadLibrary("./libpubeval.so")
lib.pubeval.restype = c_float
intp = ctypes.POINTER(ctypes.c_int32)
lib.pubeval.argtypes = [ctypes.c_int32, intp]


def is_race(board: np.ndarray):
    comp_pos = np.where(board[1:26] > 0)[0]
    opp_pos = np.where(board[0:25] < 0)[0]
    if len(comp_pos) == 0 or len(opp_pos) == 0:
        return 1
    return 1 if comp_pos[-1] < opp_pos[0] else 0


def pubeval(board: np.ndarray):
    board = board.astype(np.int32)
    race = is_race(board)
    score = lib.pubeval(c_int(race), board.ctypes.data_as(intp))
    return score
