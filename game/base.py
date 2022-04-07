from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Hashable, Iterable, List, Tuple

import numpy as np
from torch.utils.data import Dataset


class Player:
    @abstractmethod
    def play(self, game: "Game") -> "Action":
        pass


class Result(IntEnum):
    WIN = 1
    LOSS = -1
    DRAW = 0


class GameState(ABC):
    """Class to represent a games state"""
    current_player: int
    board: np.ndarray

    @abstractmethod
    def hash(self) -> Hashable:
        pass


class Action(ABC):
    @abstractmethod
    def __call__(self, state: GameState) -> GameState:
        pass


class GameConfig(ABC):
    nrow: int
    ncol: int
    start_player: int
    pass


class InvalidActionError(Exception):
    pass


class Game(ABC):
    """Abstract Class to represent interface to turn-based games"""

    cfg: GameConfig
    state: GameState

    def __init__(self, cfg: GameConfig) -> None:
        self.cfg = cfg
        self._state: GameState

    @abstractmethod
    def get_action_space(self) -> int:
        """Return the (max) number of possible actions in a game"""
        pass

    @abstractmethod
    def get_valid_actions(self) -> List[Tuple[int, "Action"]]:
        """
        Return list legal actions based on current state

        First element of the tuple is the index of the action in the (flattened)
        action space, the second is the Action itself
        """
        pass

    @abstractmethod
    def get_action(self, idx: int) -> Action:
        """Return the action represented by that idx in the (flat) action space"""
        pass

    @abstractmethod
    def move(self, action: Action) -> "Game":
        """Perform a move and return a Game with the next state"""
        pass

    @property
    @abstractmethod
    def over(self) -> bool:
        """Has game reached an end-scenario?"""
        pass

    @abstractmethod
    def reward_player(self, player: int) -> Result:
        """
        Returns a Result (1, 0, -1) corresponding to a win, draw or loss

        Defaults to rewarding the current player
        """
        pass

    @abstractmethod
    def encode(self) -> np.ndarray:
        """Encode current game into NN representation"""
        pass

    @abstractmethod
    def decode(self, encoded: np.ndarray):
        """Decode NN representation"""
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass


class GamePlayDataset(Dataset[Tuple[np.ndarray, np.ndarray, float]]):
    def __init__(
        self,
        data: Iterable[Tuple[np.ndarray, np.ndarray, float]],
        *,
        flip_h: bool = False,
        flip_v: bool = False,
        flip_hv: bool = False,
    ) -> None:
        """
        Convert a list of tuples of boards, policy arrays and values into a Dataset
        """
        self.data = list(data)
        for cond, ax in zip((flip_v, flip_h, flip_hv), (0, 1, (0, 1))):
            if cond:
                self.data += [
                    (np.flip(s, axis=ax), np.flip(p, axis=ax), v) for s, p, v in data
                ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]
