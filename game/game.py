from abc import ABC, abstractmethod, abstractproperty
from enum import IntEnum
from typing import Iterable, List, Optional, Tuple, Union

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
    pass


class Game(ABC):
    """Abstract Class to represent interface to turn-based games"""

    cfg: GameConfig
    state: GameState

    def __init__(self, cfg: GameConfig) -> None:
        self.cfg = cfg
        self.state: GameState

    @abstractmethod
    def get_action_space(self) -> int:
        """Return the (max) number of possible actions in a game"""
        pass

    @abstractmethod
    def get_valid_actions(self) -> List[Union["Action", None]]:
        """Return list of legal actions based on current state"""
        pass

    @abstractmethod
    def move(self, action: Action) -> "Game":
        """Perform a move and return a Game with the next state"""
        pass

    @abstractproperty
    def over(self) -> bool:
        """Has game reached an end-scenario?"""
        pass

    @abstractmethod
    def reward_player(self, player: int) -> Optional[Result]:
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

    @abstractmethod
    def __repr__(self) -> str:
        pass


class GamePlayDataset(Dataset):
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

    def __getitem__(self, idx):
        return self.data[idx]
