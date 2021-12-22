from abc import ABC, abstractclassmethod, abstractmethod
from enum import IntEnum
from typing import List, Optional

import numpy as np


class Result(IntEnum):
    WIN = 1
    LOSS = -1
    DRAW = 0


class GameState(ABC):
    pass


class Action(ABC):
    @abstractmethod
    def __call__(self, state: GameState) -> GameState:
        pass


class Game(ABC):
    """Abstract Class to represent interface to turn based games"""

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
