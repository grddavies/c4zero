import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from c4zero import C4Zero
from game.game import Game, GamePlayDataset, GameState
from MCTS import MCTS


class Trainer:
    def __init__(self, game: Game, model: C4Zero, args: Dict):
        self.model = model
        self.game = game
        self.args = args
        self.mcts = MCTS(game, self.model)

    def execute_episode(
        self, n_simulations: int = 100, t_mcts: float = 0.0, t_move_thresh: int = 0
    ):
        """
        Run through an entire game of self play

        Parameters
        ----------
        n_simulations: int (default = 100)
            The number of simulations per move during MCTS.

        t_mtcs: float (default = 0)
            The temperature parameter to be used for the first `t_move_thresh` moves
            during MCTS. This parameter controls the degree of exploration (higher t,
            more exploration behaviour).#

        t_move_thresh: int (default = 0)
            The number of moves after which the temperature parameter is set to zero.
            After this threshold, nodes are selected by max visit count.

        Returns
        -------
        training_examples: List[Tuple[GameState, np.array, int]]
            A list of training data that links the GameState (a board and
            current player) with the action probabilities found by MCTS and
            the reward value from the perspective of the current player.
        """
        # Start MCTS from init game
        self.mcts = MCTS(self.game, self.model)
        # Match game states and rewards?
        train_examples: List[Tuple[GameState, np.ndarray]] = []
        n_moves = 0
        while True:
            # Run n simultaions from current state
            root = self.mcts.run(n_simulations)
            action_probs = [0 for _ in range(self.game.get_action_space())]
            # Action probs are the n visits to each node
            for k, v in root.children.items():
                action_probs[k] = v.n
            # Normalise action probabilites
            action_probs = action_probs / np.sum(action_probs)
            train_examples.append((root.game.state, action_probs))
            if n_moves > t_move_thresh:
                tau = t_mcts
            else:
                tau = 0
            action_index = root.select_action(temperature=tau)
            action = root.game.get_valid_actions()[action_index]
            # Update the starting state in MCTS
            self.mcts.game = self.mcts.game.move(action)
            n_moves += 1

            if not self.mcts.game.over:
                continue
            # Reward from the perspective of the current player at game end
            reward = self.mcts.game.reward_player()
            current_player = self.mcts.game.state.current_player
            # We return the board state, action probs from MCTS and the reward
            # from the current player's perspective
            return [
                (
                    h_state.board,
                    h_action_probs,
                    reward * (-1) ** (h_state.current_player != current_player),
                )
                for h_state, h_action_probs in train_examples
            ]

    def learn(self, n_iters: int = 500, n_eps: int = 100):
        """
        Generate training data by self-play and train neural net on these data.

        Parameters
        ----------
        n_iters: int (default = 500)
            Total number of training iterations.

        """
        for i in range(1, n_iters + 1):
            # TODO: NNet comparison during training
            print(f"[{i}/{n_iters}]")
            traindata = GamePlayDataset(
                sum((self.execute_episode() for _ in range(n_eps)), [])
            )
            self.train(traindata)
        return self

    def train(self, traindata: GamePlayDataset, batch_size: int = 64):
        # TODO: add lr scheduler
        optimizer = optim.Adam(self.model.parameters(), lr=5e-4)
        p_losses = []
        v_losses = []
        dataloader = DataLoader(traindata, batch_size=batch_size, shuffle=True)
        self.model.train()

        for epoch in range(self.args["epochs"]):
            for i, data in enumerate(dataloader, 0):
                boards, target_ps, target_vs = data
                # FIXME: only works for 1d boards
                boards = boards.squeeze().float()
                target_ps = target_ps.float()
                target_vs = target_vs.float()

                # compute output
                out_pi, out_v = self.model(boards)
                l_pi = self.loss_pi(target_ps, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                p_losses.append(float(l_pi))
                v_losses.append(float(l_v))

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            print(
                f"\t[{epoch:d}, {batch_size:2d}]",
                f"Policy Loss: {np.mean(p_losses):.6f}",
                f"Value Loss: {np.mean(v_losses):.6f}",
                sep="\n\t",
            )

    def loss_pi(self, targets, outputs) -> torch.FloatTensor:
        loss = -(targets * torch.log(outputs)).sum(dim=1)
        return loss.mean()

    def loss_v(self, targets, outputs) -> torch.FloatTensor:
        loss = torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]
        return loss

    def save_checkpoint(self, folder, filename):
        if not os.path.exists(folder):
            os.mkdir(folder)
        filepath = os.path.join(folder, filename)
        torch.save({"state_dict": self.model.state_dict()}, filepath)
