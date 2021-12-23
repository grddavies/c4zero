from math import sqrt
from typing import Dict, List
import numpy as np

from game.game import Game, Player


class Network:
    """DummyNetwork for standard connect 4 game"""
    def predict(self, state_rep):
        return np.array([1 / 7 for _ in range(7)]), 0.5


class Node:
    def __init__(self, prior: float, game: Game):
        self.prior = prior  # prob of selecting node
        self.value_sum = 0  # total value from all visits
        self.children: Dict[int, "Node"] = {}  # legal child positions
        self.game = game
        self._visit_count = 0

    @property
    def n(self):
        return self._visit_count

    @property
    def expanded(self):
        return len(self.children) > 0

    @property
    def value(self):
        if self.n == 0:
            return 0
        return self.value_sum / self.n

    def select_action(self, temperature):
        """
        Select an action from this node

        Actions are chosen according to the visit count distribution and temperature.
        """
        visit_counts = np.array([child.n for child in self.children.values()])
        actions = [action for action in self.children.keys()]
        if temperature == 0:
            action = actions[np.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = np.random.choice(actions)
        else:
            # See paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution /= sum(visit_count_distribution)
            action = np.random.choice(actions, p=visit_count_distribution)
        return action

    def select_child(self):
        """Select the child with the highest UCB score"""
        best_score = -np.inf
        best_child = None
        for child in self.children.values():
            score = ucb_score(self, child)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def expand(self, action_probs: np.ndarray):
        """
        Expand a node and keep track of the prior policy probability
        """
        actions = self.game.get_valid_actions()
        for a, prob in enumerate(action_probs):
            # NOTE: set invalid move probs to zero
            if prob == 0:
                continue
            self.children[a] = Node(prob, self.game.move(actions[a]))

    def __repr__(self):
        """Pretty print node info"""
        p = "{0:.2f}".format(self.prior)
        return f"{self.game}\nPrior: {p} Count: {self.n} Value: {self.value}"


def ucb_score(parent: Node, child: Node):
    """Calculate the upper confidence bound score between nodes"""
    prior_score = child.prior * sqrt(parent.n) / (child.n + 1)
    if child.n > 0:
        # The value of the child is from the perspective of the opposing player
        value_score = -child.value
    else:
        value_score = 0
    return value_score + prior_score


class MCTS:
    """Class to perform a Monte Carlo Tree Search"""

    def __init__(self, game: Game, model: Network):
        self.game = game
        self.model = model

    def run(self, n_simulations):
        """Run Monte Carlo Tree Search"""

        root = Node(0, self.game)
        # EXPAND root
        # Get policy, value from model
        action_probs, value = self.model.predict(root.game.encode())
        valid_moves = self.game.get_valid_actions()
        action_probs = self.normalize_probabilities(action_probs, valid_moves)
        root.expand(action_probs)

        for _ in range(n_simulations):
            node = root
            search_path: List[Node] = [node]

            # SELECT
            while node.expanded:
                node = node.select_child()
                search_path.append(node)

            if not node.game.game_over:
                # EXPAND node
                action_probs, value = self.model.predict(node.game.encode())
                valid_moves = node.game.get_valid_actions()
                action_probs = self.normalize_probabilities(action_probs, valid_moves)
                node.expand(action_probs)

            self.backpropagate(search_path, value, node.game.state.current_player)

        return root

    def backpropagate(
        self, search_path: List[Node], value: float, current_player: Player
    ):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root.
        """
        for node in reversed(search_path):
            node.value_sum += (
                value if node.game.state.current_player == current_player else -value
            )
            node._visit_count += 1

    @staticmethod
    def normalize_probabilities(policy, actions):
        valid_actions = [action is not None for action in actions]
        policy = policy * valid_actions  # Mask invalid moves
        policy /= np.sum(policy)  # Normalise new probs
        return policy
