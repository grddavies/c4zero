import copy
from enum import IntEnum
from typing import List

import numpy as np

from game.game import Action, Game, GameState, Result


class ConnectPlayer(IntEnum):
    RED = 1
    YELLOW = -1


class ConnectGameState(GameState):
    def __init__(self, board: np.ndarray, current_player: ConnectPlayer):
        self.board = board
        self.current_player = current_player


class ConnectAction(Action):
    """Actions in a connect-X style game"""

    def __init__(self, col: int):
        self.col = col

    def __call__(self, state: ConnectGameState):
        new_board = state.board.copy()
        if new_board[0][self.col] != 0:
            raise UserWarning("Invalid move")
        # Get row to drop piece into
        row = new_board[:, self.col].nonzero()[0]
        if row.size > 0:
            row = row.max()
        else:
            row, _ = new_board.shape
        # Place piece
        new_board[row - 1, self.col] = state.current_player
        return ConnectGameState(new_board, -state.current_player)


class ConnectGame(Game):
    def __init__(
        self, nrow: int, ncol: int, win_cond: int = 4, start_player: ConnectPlayer = 1
    ):
        self.nrow = nrow
        self.ncol = ncol
        self.win_cond = win_cond
        if win_cond > max(nrow, ncol):
            raise ValueError(
                f"Win condition {win_cond} too long for ",
                f"board shape ({nrow}, {ncol})",
            )
        self.state = ConnectGameState(np.zeros((nrow, ncol), dtype=int), start_player)

    @property
    def state(self) -> ConnectGameState:
        return self._state

    @state.setter
    def state(self, state: GameState):
        if state.board.shape != (self.nrow, self.ncol):
            raise ValueError("Incorrect board dimensions for this game")
        self._state = state

    def move(self, col: int, action: Action = None):
        if action is None:
            action = ConnectAction(col)
        newgame = copy.deepcopy(self)
        newgame.state = action(self.state)
        return newgame

    def game_result(self):
        """Returns a Result (1, 0, -1) corresponding to a win, draw or loss"""
        if self.check_winner(1):
            return Result(1)
        if self.check_winner(-1):
            return Result(-1)
        if self.get_action_space() is []:
            return Result(0)

    def get_action_space(self) -> List[Action]:
        return [i for i, x in enumerate(self.state.board[0, :]) if x == 0]

    def _h_check(self, p: ConnectPlayer, n: int):
        """Check for n pieces belonging to p in a horizontal line"""
        for i in range(self.nrow):
            for j in range(self.ncol - (n - 1)):
                if all(self.state.board[i, j + x] == p for x in range(n)):
                    return True

    def _v_check(self, p: ConnectPlayer, n: int):
        """Check for n pieces belonging to p in a vertical line"""
        for i in range(self.nrow - (n - 1)):
            for j in range(self.ncol):
                if all(self.state.board[i + x, j] == p for x in range(n)):
                    return True

    def _d_check(self, p: ConnectPlayer, n: int):
        """Check for n pieces belonging to p in a diagonal line"""
        # Ascending diagonal check
        for i in range(self.nrow - (n - 1)):
            for j in range(n - 1, self.ncol):
                if all(self.state.board[i + x, j - x] == p for x in range(n)):
                    return True
        # Descending diagonal check
        for i in range(self.nrow - (n - 1)):
            for j in range(self.ncol - (n - 1)):
                if all(self.state.board[i + x, j + x] == p for x in range(n)):
                    return True

    def check_winner(self, p: ConnectPlayer):
        """Check if player p has won"""
        # Horizontal check
        if self._h_check(p, self.win_cond):
            return True
        # Vertical check
        if self._v_check(p, self.win_cond):
            return True
        # Diagonals check
        if self._d_check(p, self.win_cond):
            return True

    @property
    def game_over(self) -> bool:
        return any(
            (
                self.get_action_space() is [],
                self.check_winner(1),
                self.check_winner(-1),
            )
        )
