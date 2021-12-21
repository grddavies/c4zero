import numpy as np


def encode_board(board: np.ndarray, current_player: int):
    nrow, ncol = board.shape
    encoded = np.zeros([nrow, ncol, 3], dtype=int)
    encoded[:, :, 0] = board == 1
    encoded[:, :, 1] = board == -1
    encoded[:, :, 2] = current_player
    return encoded.astype(int)


def decode_board(encoded: np.ndarray):
    decoded = encoded[:, :, 0] - encoded[:, :, 1]
    current_player = decoded[0, 0, 2]
    return decoded, current_player
