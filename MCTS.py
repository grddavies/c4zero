from collections import defaultdict
from typing import List, Optional

import numpy as np

from game.game import Action, Game, Result


class MCTSNode:
    """MonteCarlo Tree Search Node Class"""

    def __init__(
        self,
        state: Game,
        parent: Optional["MCTSNode"] = None,
        parent_action: Action = None,
    ):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.children: List["MCTSNode"] = []
        self._number_of_visits = 0
        self._results = defaultdict(int)
        self._results[1] = 0
        self._results[-1] = 0
        self._untried_actions = None
        self._untried_actions = self.untried_actions

    @property
    def untried_actions(self) -> List[Action]:
        """Get list of legal moves to be taken from current state"""
        self._untried_actions = self.state.get_action_space()
        return self._untried_actions

    @property
    def q(self):
        """Return the difference between wins and losses from current state"""
        wins = self._results[1]
        loses = self._results[-1]
        return wins - loses

    @property
    def n(self):
        """Return number of visits to this node"""
        return self._number_of_visits

    def expand(self):
        action = self._untried_actions.pop()
        next_state = self.state.move(action)
        child_node = MCTSNode(next_state, parent=self, parent_action=action)
        self.children.append(child_node)
        return child_node

    @property
    def terminal_node(self) -> bool:
        return self.state.game_over

    def rollout(self):
        current_rollout_state = self.state
        while not current_rollout_state.game_over:
            possible_moves = current_rollout_state.get_action_space()
            action = self.rollout_policy(possible_moves)
            current_rollout_state = current_rollout_state.move(action)
        return current_rollout_state.game_result()

    def backpropagate(self, result: Result):
        self._number_of_visits += 1.0
        self._results[result] += 1.0
        if self.parent:
            self.parent.backpropagate(result)

    @property
    def fully_expanded(self) -> bool:
        return len(self._untried_actions) == 0

    def best_child(self, c_param=0.1):
        """Select the best child node from the children array"""
        choices_weights = [
            (c.q / c.n) + c_param * np.sqrt((2 * np.log(self.n) / c.n))
            for c in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, possible_moves: List[Action]):
        """Randomly select a move to play out"""
        return possible_moves[np.random.randint(len(possible_moves))]

    def _tree_policy(self):
        """Select node to run rollout"""
        current_node = self
        while not current_node.terminal_node:
            if not current_node.fully_expanded:
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node

    def best_action(self, n_iters):
        for _ in range(n_iters):
            v = self._tree_policy()
            reward = v.rollout()
            v.backpropagate(reward)
        return self.best_child(c_param=0.0)
