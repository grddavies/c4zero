# pyright: reportUnknownMemberType=false
import numpy as np

from . import players
from .connect import ConnectGame, ConnectGameConfig, DropPiece

c3 = ConnectGame(ConnectGameConfig(3, 3, 3))


class TestGreedy:
    player = players.Greedy()

    def test_takes_winning_moves(self):
        cases = [
            # Vertical
            (np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]]), 0),
            # Horizontal
            (np.array([[0, 0, 0], [0, 0, 0], [1, 0, 1]]), 1),
            # Asc Diag
            (np.array([[0, 0, 0], [0, 1, -1], [1, -1, -1]]), 2),
            # Desc Diag
            (np.array([[1, 0, 0], [-1, 0, 0], [1, -1, 1]]), 1),
        ]
        for board, expected in cases:
            c3.state.board = board
            action = self.player.play(c3)
            assert isinstance(action, DropPiece)
            assert action.col == expected

    def test_blocking_opponent(self):
        cases = [
            # Vertical
            (np.array([[0, 0, 0], [-1, 0, 0], [-1, 0, 0]]), 0),
            # Horizontal
            (np.array([[0, 0, 0], [0, 0, 0], [-1, 0, -1]]), 1),
            # Asc Diag
            (np.array([[0, 0, 0], [0, -1, 1], [-1, 1, -1]]), 2),
            # Desc Diag
            (np.array([[-1, 0, 0], [1, 0, 0], [-1, 1, -1]]), 1),
        ]
        for board, expected in cases:
            c3.state.board = board
            action = self.player.play(c3)
            assert isinstance(action, DropPiece)
            assert action.col == expected

    def test_prefers_winning(self):
        cases = [
            # Vertical
            (np.array([[0, 0, 0], [-1, 1, 0], [-1, 1, 0]]), 1),
            # NOTE: No other cases valid
        ]
        for board, expected in cases:
            c3.state.board = board
            action = self.player.play(c3)
            assert isinstance(action, DropPiece)
            assert action.col == expected
