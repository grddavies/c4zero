import numpy as np
import unittest

import pytest
from MCTS import Node, MCTS
from game.connect.connect import ConnectGameConfig, ConnectGame

connect2 = ConnectGameConfig(1, 4, 2)


class TestMCTS:

    def test_mcts_from_root_with_equal_priors(self):
        class MockModel:
            def predict(self, _):
                return np.array([0.25, 0.25, 0.25, 0.25]), 0.5

        model = MockModel()
        mcts = MCTS(ConnectGame(connect2), model)
        root = mcts.run(50)

        # the best move is to play at index 1 or 2
        best_outer_move = max(root.children[0].n, root.children[3].n)
        best_center_move = max(root.children[1].n, root.children[2].n)
        assert best_center_move > best_outer_move

    def test_mcts_finds_best_move_with_really_bad_priors(self):
        class MockModel:
            def predict(self, _):
                return np.array([0.3, 0.7, 0, 0]), 0.0001

        model = MockModel()
        mcts = MCTS(ConnectGame(connect2), model)
        mcts.game.state.board = np.array([[0, 0, 1, -1]])
        print("starting")
        root = mcts.run(25)
        print(root)
        # the best move is to play at index 1
        assert root.children[1].n > root.children[0].n

    def test_mcts_finds_best_move_with_equal_priors(self):

        class MockModel:
            def predict(self, board):
                return np.array([0.51, 0.49, 0, 0]), 0.0001

        model = MockModel()
        mcts = MCTS(ConnectGame(connect2), model)
        mcts.game.state.board = np.array([[0, 0, -1, 1]])
        root = mcts.run(25)

        # the better move is to play at index 1
        root.children[0].n < root.children[1].n

    def test_mcts_finds_best_move_with_really_really_bad_priors(self):
        class MockModel:
            def predict(self, board):
                return np.array([0, 0.3, 0.3, 0.3]), 0.0001
        model = MockModel()
        mcts = MCTS(ConnectGame(connect2), model)
        mcts.game.state.board = np.array([[-1, 0, 0, 0]])
        root = mcts.run(100)

        # the best move is to play at index 1
        root.children[1].n > root.children[2].n
        root.children[1].n > root.children[3].n


class NodeTests(unittest.TestCase):

    def test_initialization(self):
        node = Node(0.5, ConnectGame(connect2))

        assert node.n == 0
        assert node.prior == 0.5
        assert len(node.children) == 0
        assert not node.expanded
        assert node.value == 0

    def test_selection(self):
        node = Node(0.5, ConnectGame(connect2))
        c0 = Node(0.5, ConnectGame(connect2))
        c1 = Node(0.5, ConnectGame(connect2))
        c2 = Node(0.5, ConnectGame(connect2))
        node._visit_count = 1
        c0._visit_count = 0
        c2._visit_count = 0
        c2._visit_count = 1

        node.children = {
            0: c0,
            1: c1,
            2: c2,
        }

        action = node.select_action(temperature=0)
        self.assertEqual(action, 2)

    def test_expansion(self):
        node = Node(0.5, ConnectGame(connect2))
        action_probs = np.array([0.25, 0.15, 0.5, 0.1])

        node.expand(action_probs)

        assert len(node.children) == 4
        assert node.expanded
        assert node.game.state.current_player == 1
        assert node.children[0].prior == 0.25
        assert node.children[1].prior == 0.15
        assert node.children[2].prior == 0.50
        assert node.children[3].prior == 0.10

    @pytest.mark.skip("pb_c and pb_init change expected results")
    def test_ucb_score_no_children_visited(self):
        node = Node(0.5, ConnectGame(connect2))
        node._visit_count = 1  # type: ignore
        action_probs = np.array([0.25, 0.15, 0.5, 0.1])  # type: ignore

        node.expand(action_probs)
        node.children[0]._visit_count = 0  # type: ignore
        node.children[1]._visit_count = 0  # type: ignore
        node.children[2]._visit_count = 0  # type: ignore
        node.children[3]._visit_count = 0  # type: ignore

        score_0 = node.ucb_score(node.children[0])
        score_1 = node.ucb_score(node.children[1])
        score_2 = node.ucb_score(node.children[2])
        score_3 = node.ucb_score(node.children[3])

        # With no visits, UCB score is just the priors
        assert score_0 == node.children[0].prior
        assert score_1 == node.children[1].prior
        assert score_2 == node.children[2].prior
        assert score_3 == node.children[3].prior

    @pytest.mark.skip("pb_c and pb_init change expected results")
    def test_ucb_score_one_child_visited(self):
        node = Node(0.5, ConnectGame(connect2))
        node._visit_count = 1  # type: ignore

        action_probs = np.array([0.25, 0.15, 0.5, 0.1])

        node.expand(action_probs)
        node.children[0]._visit_count = 0  # type: ignore
        node.children[1]._visit_count = 0  # type: ignore
        node.children[2]._visit_count = 1  # type: ignore
        node.children[3]._visit_count = 0  # type: ignore

        score_0 = node.ucb_score(node.children[0])
        score_1 = node.ucb_score(node.children[1])
        score_2 = node.ucb_score(node.children[2])
        score_3 = node.ucb_score(node.children[3])

        # With no visits, UCB score is just the priors
        assert score_0 == node.children[0].prior
        assert score_1 == node.children[1].prior
        assert score_3 == node.children[3].prior
        # If we visit one child once, its score is halved
        assert score_2 == node.children[2].prior / 2

        child = node.select_child()

        assert child == node.children[0]

    # def test_ucb_score_one_child_visited_twice(self):
    #     node = Node(0.5, ConnectGame(connect2))
    #     node._visit_count = 2
    #     action_probs = np.array([0.25, 0.15, 0.5, 0.1])

    #     node.expand(action_probs)
    #     node.children[0]._visit_count = 0
    #     node.children[1]._visit_count = 0
    #     node.children[2]._visit_count = 2
    #     node.children[3]._visit_count = 0

    #     action, child = node.select_child()

    #     # Now that we've visited the second action twice, we should
    #     # end up trying the first action
    #     self.assertEqual(action, 0)

    # def test_ucb_score_no_children_visited(self):
    #     node = Node(0.5, to_play=1)
    #     node.visit_count = 1

    #     state = [0, 0, 0, 0]
    #     action_probs = [0.25, 0.15, 0.5, 0.1]
    #     to_play = 1

    #     node.expand(state, to_play, action_probs)
    #     node.children[0].visit_count = 0
    #     node.children[1].visit_count = 0
    #     node.children[2].visit_count = 1
    #     node.children[3].visit_count = 0

    #     score_0 = ucb_score(node, node.children[0])
    #     score_1 = ucb_score(node, node.children[1])
    #     score_2 = ucb_score(node, node.children[2])
    #     score_3 = ucb_score(node, node.children[3])

    #     # With no visits, UCB score is just the priors
    #     self.assertEqual(score_0, node.children[0].prior)
    #     self.assertEqual(score_1, node.children[1].prior)
    #     # If we visit one child once, its score is halved
    #     self.assertEqual(score_2, node.children[2].prior / 2)
    #     self.assertEqual(score_3, node.children[3].prior)
