from enum import IntEnum
from typing import List

import numpy as np
from game.game import Action, Game, GameConfig, GameState, Player, Result


class ConnectPlayer(Player, IntEnum):
    RED = 1
    YELLOW = -1


class ConnectGameState(GameState):
    def __init__(self, board: np.ndarray, current_player: ConnectPlayer):
        self.board = board
        self.current_player = current_player

    def __repr__(self) -> str:
        return f"{self.board}\nCurrent player: {self.current_player}"


class DropPiece(Action):
    """Drop a piece into a column"""

    def __init__(self, col: int):
        self.col = col

    def __call__(self, state: ConnectGameState):
        board = state.board.copy()
        if board[0][self.col] != 0:
            raise UserWarning("Invalid move")
        # Get row to drop piece into
        row = board[:, self.col].nonzero()[0]
        if row.size > 0:
            row = row.min()
        else:
            row, _ = board.shape
        # Place piece - always a 1 as we always play from current player's
        # perspective
        board[row - 1, self.col] = 1
        # Invert board representation
        return ConnectGameState(-board, -state.current_player)

    def __repr__(self) -> str:
        return f"DropPiece({self.col})"


class ConnectGameConfig(GameConfig):
    def __init__(
        self,
        nrow: int = 6,
        ncol: int = 7,
        win_cond: int = 4,
        start_player: ConnectPlayer = ConnectPlayer(1),
    ):
        self.nrow, self.ncol = nrow, ncol
        if win_cond > max(nrow, ncol):
            raise ValueError(
                f"Win condition {win_cond} too long for ",
                f"board shape ({nrow}, {ncol})",
            )
        self.win_cond = win_cond
        self.start_player = start_player


class ConnectGame(Game):
    def __init__(self, cfg: ConnectGameConfig = ConnectGameConfig()):
        self.cfg = cfg
        self.state = ConnectGameState(
            np.zeros((cfg.nrow, cfg.ncol), dtype=int), cfg.start_player,
        )

    @property
    def state(self) -> ConnectGameState:
        return self._state

    @state.setter
    def state(self, state: ConnectGameState):
        if state.board.shape != (self.cfg.nrow, self.cfg.ncol):
            raise ValueError("Incorrect board dimensions for this game")
        self._state = state

    def move(self, action: Action):
        newgame = ConnectGame(self.cfg)
        newgame.state = action(self.state)
        return newgame

    def reward_player(self, player: ConnectPlayer = None):
        """Returns a Result (1, 0, -1) corresponding to a win, draw or loss"""
        if player is None:
            player = self.state.current_player
        if self.check_winner(player):
            return Result(1)
        if self.check_winner(-player):
            return Result(-1)
        if all(x is None for x in self.get_valid_actions()):
            # No winner and no remaining actions = draw
            return Result(0)

    def get_action_space(self):
        return self.cfg.ncol

    def get_valid_actions(self) -> List[Action]:
        return [
            DropPiece(i) if x == 0 else None
            for i, x in enumerate(self.state.board[0, :])
        ]

    def _h_check(self, p: ConnectPlayer, n: int):
        """Check for n pieces belonging to p in a horizontal line"""
        for i in range(self.cfg.nrow):
            for j in range(self.cfg.ncol - (n - 1)):
                if all(self.state.board[i, j + x] == p for x in range(n)):
                    return True

    def _v_check(self, p: ConnectPlayer, n: int):
        """Check for n pieces belonging to p in a vertical line"""
        for i in range(self.cfg.nrow - (n - 1)):
            for j in range(self.cfg.ncol):
                if all(self.state.board[i + x, j] == p for x in range(n)):
                    return True

    def _d_check(self, p: ConnectPlayer, n: int):
        """Check for n pieces belonging to p in a diagonal line"""
        # Ascending diagonal check
        for i in range(self.cfg.nrow - (n - 1)):
            for j in range(n - 1, self.cfg.ncol):
                if all(self.state.board[i + x, j - x] == p for x in range(n)):
                    return True
        # Descending diagonal check
        for i in range(self.cfg.nrow - (n - 1)):
            for j in range(self.cfg.ncol - (n - 1)):
                if all(self.state.board[i + x, j + x] == p for x in range(n)):
                    return True

    def check_winner(self, p: ConnectPlayer):
        """Check if player p has won"""
        if self.state.current_player != p:
            p = -1
        else:
            p = 1
        # Horizontal check
        if self._h_check(p, self.cfg.win_cond):
            return True
        # Vertical check
        elif self._v_check(p, self.cfg.win_cond):
            return True
        # Diagonals check
        elif self._d_check(p, self.cfg.win_cond):
            return True
        else:
            return False

    @property
    def over(self) -> bool:
        return any(
            (
                all(x is None for x in self.get_valid_actions()),
                self.check_winner(1),
                self.check_winner(-1),
            )
        )

    def encode(self):
        # encoded = np.zeros([self.cfg.nrow, self.cfg.ncol, 3], dtype=int)
        # encoded[:, :, 0] = self.state.board == 1
        # encoded[:, :, 1] = self.state.board == -1
        # encoded[:, :, 2] = self.state.current_player
        encoded = self.state.board
        return encoded.reshape([1, 1, self.cfg.nrow, self.cfg.ncol])

    def decode(self, encoded: np.ndarray):
        # self.state.board = encoded[:, :, 0] - encoded[:, :, 1]
        # self.state.current_player = encoded[0, 0, 2]
        pass

    def __repr__(self):
        return f"{self.state}\nWin condition: {self.cfg.win_cond}"
