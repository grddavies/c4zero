from enum import IntEnum
import numpy as np


class Player(IntEnum):
    RED = 1
    YELLOW = -1


class Board:
    def __init__(self, nrow: int, ncol: int):
        self.nrow = nrow
        self.ncol = ncol
        self.init_board = np.zeros((nrow, ncol), dtype=int)
        self.current_board = self.init_board
        self.current_player = Player(1)

    def _next_player(self):
        self.current_player = Player(-self.current_player)

    def move(self, col: int):
        if col > self.ncol:
            raise ValueError("Column index out of bounds")
        if self.current_board[0][col] != 0:
            raise UserWarning("Invalid move")
        # Get row to drop piece into
        row = self.current_board[:, col].nonzero()[0]
        if row.size > 0:
            row = row.max()
        else:
            row = self.nrow
        # Place piece
        self.current_board[row - 1, col] = self.current_player
        # Change player
        self._next_player()

    @staticmethod
    def check_winner(board: np.ndarray, p: Player, n: int):
        nrow, ncol = board.shape
        # Horizontal check
        for i in range(nrow):
            for j in range(ncol - (n - 1)):
                if all([board[i, j + x] == p for x in range(n)]):
                    return True
        # Vertical check
        for i in range(nrow - (n - 1)):
            for j in range(ncol):
                if all([board[i + x, j] == p for x in range(n)]):
                    return True
        # Ascending diagonal check
        for i in range(nrow - (n - 1)):
            for j in range(n-1, ncol):
                if all([board[i + x, j - x] == p for x in range(n)]):
                    return True
        # Descending diagonal check
        for i in range(nrow - (n - 1)):
            for j in range(ncol - (n - 1)):
                if all([board[i + x, j + x] == p for x in range(n)]):
                    return True

    def get_winner(self, p: Player, n: int = 4):
        if self.checkwinner(self.current_board, 1, n):
            return Player(1)
        elif self.checkwinner(self.current_board, -1, n):
            return Player(-1)
        else:
            return None

    def get_legal_moves(self):
        return np.nonzero(self.current_board[0, :] == 0)[0]
