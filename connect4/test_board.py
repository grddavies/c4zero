from . import Board
import numpy as np


class TestBoard:
    def test_init_board(self):
        # nrow, ncol, expected
        cases = [
            ((0, 0), (0, 0)),
            ((1, 1), (1, 1)),
            ((6, 7), (6, 7)),
            ((99, 10), (99, 10))
        ]
        for (nrow, ncol), expected in cases:
            board = Board(nrow, ncol)
            assert board.init_board.shape == expected
            assert np.sum(board.init_board) == 0
            assert board.current_player == 1

    def test_move(self):
        board = Board(6, 7)
        board.move(col=0)
        assert board.current_board[5, 0] == 1
        assert board.current_player == -1
        board.move(0)
        assert board.current_board[5, 0] == 1
        assert board.current_board[4, 0] == -1
        assert board.current_player == 1

    def test_check_winner(self):
        # horizontal
        board = Board(6, 7)
        board.current_board[-1, -4:] = -1
        assert Board.check_winner(board.current_board, -1, 4)
        assert not Board.check_winner(board.current_board, 1, 4)
        assert not Board.check_winner(board.current_board, -1, 5)

        # vertical
        board = Board(6, 7)
        board.current_board[-4:, 0] = 1
        assert Board.check_winner(board.current_board, 1, 4)
        assert not Board.check_winner(board.current_board, -1, 4)
        assert not Board.check_winner(board.current_board, 1, 5)

        # desc diag
        board = Board(4, 4)
        board.current_board = np.identity(4)
        assert Board.check_winner(board.current_board, 1, 4)
        assert not Board.check_winner(board.current_board, -1, 4)
        assert not Board.check_winner(board.current_board, 1, 5)

        # asc diag
        board = Board(6, 7)
        for x in range(1, 5):
            board.current_board[x, -x] = 1
        assert Board.check_winner(board.current_board, 1, 4)
        assert not Board.check_winner(board.current_board, -1, 4)
        assert not Board.check_winner(board.current_board, 1, 5)
        for x in range(1, 6):
            board.current_board[x, -x] = 1
        assert Board.check_winner(board.current_board, 1, 5)
