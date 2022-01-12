from typing import Dict

import joblib
from joblib.parallel import delayed

from game.base import Game, Player
from util import ProgressParallel


class Arena:
    """
    Pit two players against each other at a game
    """

    players: Dict[int, Player]
    game: Game
    n_jobs: int

    def __init__(
        self, player_1: Player, player_2: Player, game: Game, n_jobs: int = 1
    ) -> None:
        self.players = {1: player_1, -1: player_2}
        self.game = game
        self.n_jobs = joblib.parallel.effective_n_jobs(n_jobs)

    def play_game(self):
        """
        Execute a single episode of a game

        Returns
        -------
        result: int
            +1 if player_1 won, -1 if player_2 won, 0 if draw
        """
        game = self.game
        while not game.over:
            p = self.players[game.state.current_player]
            action = p.play(game)
            game = game.move(action)
        return game.reward_player(1)

    def play_games(self, n_games: int):
        """
        Execute n_games, swapping starting player after for half the games
        """
        n_games_per_half = n_games // 2
        res1 = ProgressParallel(self.n_jobs, total=n_games_per_half, leave=False)(
            delayed(self.play_game)() for _ in range(n_games_per_half)
        )
        # Switch starting player
        self.game.cfg.start_player *= -1
        res2 = ProgressParallel(self.n_jobs, total=n_games_per_half, leave=False)(
            delayed(self.play_game)() for _ in range(n_games_per_half)
        )
        res = res1 + res2
        return res.count(1), res.count(-1), res.count(0)
