import math
from typing import Dict, Hashable, List

import numpy as np

from c4zero import C4Zero
from game.base import Game


class Node:
    pb_c_base: int = 19652
    pb_c_init: float = 1.25

    prior: float
    value_sum: float
    children: Dict[int, "Node"]
    game: Game
    _visit_count: int

    def __init__(self, prior: float, game: Game):
        self.prior = prior  # prob of selecting node
        self.value_sum = 0  # total value from all visits
        self.children = {}  # legal child positions
        self.game = game
        # TODO: remove prefix `_`
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

    def increment(self):
        """Increment visit count"""
        self._visit_count += 1
        return self

    def select_action(self, temperature: float):
        """
        Select an action from this node

        Actions are chosen according to the visit count distribution and temperature.
        """
        visit_counts = np.array([child.n for child in self.children.values()])  # type: ignore
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
        if not self.expanded:
            raise UserWarning("Node not expanded, cannot select a child")
        children_scores = [(self.ucb_score(c), c) for c in self.children.values()]
        max_score = max(score for score, _ in children_scores)
        best_child = next(c for score, c in children_scores if score == max_score)
        return best_child

    def expand(self, action_probs: np.ndarray):
        """
        Expand a node and keep track of the prior policy probability
        """
        # tuples of idx and Action
        actions = self.game.get_valid_actions()
        # Get action space and put 1s where idx in actions(a, _)
        valid_idx = [i for i, _ in actions]
        valid_actions = np.zeros(self.game.get_action_space(), dtype=int)
        valid_actions[valid_idx] = 1
        action_probs = action_probs * valid_actions  # Mask invalid moves
        action_probs /= action_probs.sum()  # Normalise new probs
        for idx, prob in enumerate(action_probs.flatten()):
            prob: float
            if prob == 0:
                continue
            action = self.game.get_action(idx)
            self.children[idx] = Node(prob, self.game.move(action))

    def ucb_score(self, child: "Node"):
        """Calculate the upper confidence bound score between nodes"""
        # Exploration bonus based on prior score
        c_puct = (
            math.log((self.n + self.pb_c_base + 1) / self.pb_c_base) + self.pb_c_init
        )
        c_puct *= math.sqrt(self.n) / (child.n + 1)
        prior_score = c_puct * child.prior
        # The value of the child is from the perspective of the opposing player
        value_score = -child.value
        return value_score + prior_score

    def add_exploration_noise(self, alpha: float = 0.3, e_frac: float = 0.25):
        """
        Add Dirichlet noise to a Node's priors to increase exploratory behaviour

        Parameters
        ----------

        alpha: float (default = 0.3)
            The shape of the gamma distribution. Must be positive.

        e_frac: float (default = 0.25)
            The exploration fraction - how much noise to apply to the priors
        """
        if not self.expanded:
            raise UserWarning("Cannot add noise to Node before expansion")
        actions = self.children.keys()
        noise = np.random.gamma(alpha, 1, len(actions))
        for a, n in zip(actions, noise):
            self.children[a].prior = self.children[a].prior * (1 - e_frac) + n * e_frac
        return self

    def __str__(self):
        """Pretty print node info"""
        p = "{0:.2f}".format(self.prior)
        return (
            f"{self.game.state}\nPrior: {p} Count: {self.n} "
            + f"Value: {self.value}\nExpanded: {self.expanded}"
        )

    def __repr__(self) -> str:
        return f"Node({self.prior}, {self.game!r})"


class MCTS:
    """Class to perform a Monte Carlo Tree Search"""

    game: Game
    model: C4Zero
    _Ps: Dict[Hashable, np.ndarray]
    _vs: Dict[Hashable, float]

    def __init__(self, game: Game, model: C4Zero):
        self.game = game
        self.model = model
        # Cache for prediction values
        self._Ps = {}
        self._vs = {}

    def run(
        self,
        n_simulations: int,
        root_dirichlet_alpha: float = 0.3,
        root_explore_frac: float = 0.25,
    ):
        """Run Monte Carlo Tree Search"""
        root = Node(0, self.game)
        # Expand root
        # Get policy, value from model
        action_probs, _ = self.cached_predict(root)

        root.expand(action_probs)
        root.add_exploration_noise(root_dirichlet_alpha, root_explore_frac)

        # Simulate gameplay from this position
        for _ in range(n_simulations):
            node = root
            search_path: List[Node] = [node]

            # Select
            while node.expanded:
                node = node.select_child()
                search_path.append(node)

            if node.game.over:
                value = node.game.reward_player(node.game.state.current_player)
            else:
                # Expand & Evaluate
                # TODO: #3 Randomly reflect/rotate board along game symmetry line here
                # See Methods: Expand and Evaluate
                action_probs, value = self.cached_predict(node)
                node.expand(action_probs)
            # Backup
            MCTS.backpropagate(search_path, value, node.game.state.current_player)

        return root

    @staticmethod
    def backpropagate(search_path: List[Node], value: float, current_player: int):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root.
        """
        for node in reversed(search_path):
            node.value_sum += (
                value if node.game.state.current_player == current_player else -value
            )
            node.increment()

    def cached_predict(self, node: Node):
        """Predict policy at a given node and cache the result"""
        s = node.game.state.hash()
        if s not in self._Ps.keys():
            self._Ps[s], self._vs[s] = self.model.predict(node.game.encode())
        return self._Ps[s], self._vs[s]
