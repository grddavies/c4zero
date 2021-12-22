from abc import ABC, abstractclassmethod, abstractmethod
from enum import IntEnum
from typing import List, Optional

import numpy as np

from game.util import EasyDict


class Result(IntEnum):
    WIN = 1
    LOSS = -1
    DRAW = 0


class GameState(ABC):
    def __init__(self, board: np.ndarray, current_player: int) -> None:
        self.board = board
        self.current_player = current_player


class Action(ABC):
    @abstractmethod
    def __call__(self, state: GameState) -> GameState:
        pass


class GameConfig(ABC, EasyDict):
    pass


class Game(ABC):
    """Abstract Class to represent interface to turn based games"""

    def __init__(self, cfg: GameConfig) -> None:
        self.cfg = cfg
        self.state: GameState

    @abstractmethod
    def get_action_space(self) -> List["Action"]:
        """Return list of legal actions based on current state"""
        pass

    @abstractmethod
    def move(self, action: Action) -> "Game":
        """Perform a move and return a Game with the next state"""
        pass

    @property
    @abstractmethod
    def game_over(self) -> bool:
        """Has game reached an end-scenario?"""
        pass

    @abstractmethod
    def game_result(self) -> Optional[Result]:
        """Returns a Result (1, 0, -1) corresponding to a win, draw or loss"""
        pass

    @abstractmethod
    def encode(self) -> np.ndarray:
        """Encode current game into NN representation"""
        pass

    @abstractmethod
    def decode(self, encoded: np.ndarray):
        """Decode NN representation"""
        pass
