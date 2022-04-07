import numpy as np
import pytest

from game.connect import ConnectGame, ConnectGameConfig, DropPiece


class TestConnectGame:
    def test_init_board(self):
        # nrow, ncol, expected
        cases = [
            ((0, 0, 0), (0, 0)),
            ((1, 1, 1), (1, 1)),
            ((6, 7, 4), (6, 7)),
            ((99, 10, 5), (99, 10))
        ]
        for (nrow, ncol, win), expected in cases:
            cfg = ConnectGameConfig(nrow, ncol, win)
            state = ConnectGame(cfg).state
            assert state.board.shape == expected
            assert np.sum(state.board) == 0  # type: ignore
            assert state.current_player == 1

    def test_move(self):
        game = ConnectGame(ConnectGameConfig(6, 7))
        assert game.state.current_player == 1
        game = game.move(DropPiece(0))
        # from p-1s perspective
        assert game.state.current_player == -1
        assert game.state.board[5, 0] == -1
        game = game.move(DropPiece(0))
        # from p1s perspective
        assert game.state.current_player == 1
        assert game.state.board[5, 0] == 1
        assert game.state.board[4, 0] == -1

    def test_win_conditions(self):
        # horizontal
        game = ConnectGame(ConnectGameConfig(6, 7, win_cond=4))
        game.state.board[-1, -4:] = -1
        assert game.over
        assert game.check_winner(-1)
        assert not game.check_winner(1)
        game.cfg.win_cond = 5
        assert not game.check_winner(-1)
        with pytest.raises(RuntimeError, match="Game not over"):
            game.reward_player(-1)
            game.reward_player(1)

        # vertical
        game = ConnectGame(ConnectGameConfig(6, 7))
        game.state.board[-4:, 0] = 1
        assert game.over
        assert game.check_winner(1)
        assert not game.check_winner(-1)
        game.cfg.win_cond = 5
        assert not game.check_winner(1)

        # desc diag
        game = ConnectGame(ConnectGameConfig(5, 5, win_cond=4))
        game.state.board[-4:, -4:] = np.identity(4)  # type: ignore
        assert game.check_winner(1)
        assert not game.check_winner(-1)
        game.cfg.win_cond = 5
        assert not game.check_winner(1)
        with pytest.raises(RuntimeError, match="Game not over"):
            game.reward_player(1)
        game.state.board[0, 0] = 1
        assert game.check_winner(1)

        # asc diag
        game = ConnectGame(ConnectGameConfig(6, 7, win_cond=4))
        for x in range(1, 5):
            game.state.board[x, -x] = 1
        assert game.check_winner(1)
        assert not game.check_winner(-1)
        game.cfg.win_cond = 5
        assert not game.check_winner(1)
        with pytest.raises(RuntimeError, match="Game not over"):
            game.reward_player(1)
        for x in range(1, 6):
            game.state.board[x, -x] = 1
        assert game.check_winner(1)

        # Draw - connect 2
        game = ConnectGame(ConnectGameConfig(1, 4, 2))
        for i in range(4):
            game.state.board[0, i] = (-1) ** i
        assert game.over
        assert not game.check_winner(1)
        assert not game.check_winner(-1)
        assert game.reward_player(1) == 0
        assert game.reward_player(-1) == 0

    def test_game_play(self):
        # Connect 2
        game = ConnectGame(ConnectGameConfig(1, 4, 2))
        game = game.move(DropPiece(0)).move(DropPiece(3)).move(DropPiece(1))
        assert game.over
        assert game.check_winner(1)
        assert game.reward_player(1) == 1
        assert game.reward_player(-1) == -1
