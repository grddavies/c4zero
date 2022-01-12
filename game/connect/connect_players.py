import numpy as np
from .connect import ConnectGame, DropPiece
from game.base import Player
from c4zero import C4Zero
from MCTS import MCTS


class HumanConnectPlayer(Player):
    @staticmethod
    def to_readable_board(board: np.ndarray):
        b = board.astype(str).squeeze()
        b = np.where(b == "1", "O", b)
        b = np.where(b == "-1", "X", b)
        b = np.where(b == "0", "-", b)
        return b

    def play(self, game: ConnectGame):
        print("Your move, you're 'O'")
        actions = game.get_valid_actions()
        while True:
            print(self.to_readable_board(game.state.board))
            try:
                selected = input(
                    "Select a column to place a piece into (zero-indexed): "
                )
                move = actions[int(selected)]
                if move is None:
                    print("Can't do that move: {selected}")
                else:
                    return move
            except (IndexError, ValueError):
                print(f"Invalid column: '{selected}'")


class RandomConnectPlayer(Player):
    verbose: bool

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose

    def play(self, game: ConnectGame):
        actions = [a for a in game.get_valid_actions() if a is not None]
        action = np.random.choice(actions)
        if self.verbose:
            if isinstance(action, DropPiece):
                print(f"Random player moved in column {action.col}")
        return action


class AIConnectPlayer(Player):
    model: C4Zero
    n_sim: int
    verbose: bool

    def __init__(self, model: C4Zero, n_sim: int = 100, verbose: bool = False):
        self.model = model
        self.n_sim = n_sim
        self.verbose = verbose

    def play(self, game: ConnectGame):
        n_moves = np.sum(game.state.board != 0) / 2
        if n_moves < 10:
            temp = 1
        else:
            temp = 0
        mcts = MCTS(game, self.model)
        if self.verbose:
            print("Thinking...")
        root = mcts.run(self.n_sim)
        move_index = root.select_action(temp)
        if self.verbose:
            print(f"AI moved in column {move_index}")
        move = game.get_valid_actions()[move_index]
        return move
