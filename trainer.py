import os
from typing import List, Tuple

import numpy as np
import torch
import torch.optim as optim
from joblib import delayed, parallel
from torch import nn
from torch.utils.data import DataLoader

from c4zero import C4Zero
from game.game import Game, GamePlayDataset, GameState
from game.util import ProgressParallel
from MCTS import MCTS


class Trainer:
    def __init__(self, game: Game, model: C4Zero, n_jobs: int = None):
        """
        Parameters
        ----------

        n_jobs: Optional int (default = None)
            Number of CPUs to use during self play. `None` means 1 unless in a
            :obj:`joblib.parallel_backend` context. `-1` means using all processors.
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.game = game
        self.mcts = MCTS(game, self.model)
        self.n_jobs = parallel.effective_n_jobs(n_jobs)

    def execute_episode(
        self, n_simulations: int = 100, t_mcts: float = 1.0, t_move_thresh: int = 8
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
            more exploration behaviour).

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

    def learn(
        self,
        n_iters: int = 500,
        n_eps: int = 100,
        epochs: int = 5,
        batch_size: int = 64,
        n_sim: int = 100,
    ):
        """
        Generate training data by self-play and train neural net on these data.

        Parameters
        ----------
        n_iters: int (default = 500)
            Total number of training iterations, self play and training.

        n_eps: int (default = 100)
            Number of games per training iteration.

        epochs: int (default = 5)
            Number of epochs per training iteration. In an epoch, all of the training
            data is used once. Epochs consister of a number of minibatches.

        batch_size: int (default = 64)
            Number of samples of training data per mini batch

        n_sim: int (default = 100)
            Number of simulations to run during MCTS
        """
        for i in range(1, n_iters + 1):
            # TODO: NNet comparison during training
            print(f"[{i}/{n_iters}]")
            print(f"\tExecuting self-play for {n_eps} episodes...")
            traindata = self.selfplay(n_eps, n_sim)
            print("\tTraining...")
            self.train(traindata, epochs=epochs, batch_size=batch_size)
        return self

    def selfplay(
        self,
        n_eps: int,
        n_sim: int = 100,
        t_mcts: float = 1.0,
        t_move_thresh: int = 8,
        flip_h: bool = False,
        flip_v: bool = False,
        flip_hv: bool = False,
    ):
        """
        Run games of self play and store output as a dataset

        n_eps: int
            Number of games to play

        n_sim: int (default = 100)
            Number of simulations to run during MCTS

        t_mtcs: float (default = 0)
            The temperature parameter to be used for the first `t_move_thresh` moves
            during MCTS. This parameter controls the degree of exploration (higher t,
            more exploration behaviour).

        t_move_thresh: int (default = 0)
            The number of moves after which the temperature parameter is set to zero.
            After this threshold, nodes are selected by max visit count.

        """
        data = ProgressParallel(self.n_jobs, total=n_eps, leave=False)(
            delayed(self.execute_episode)(n_sim, t_mcts, t_move_thresh)
            for _ in range(n_eps)
        )
        return GamePlayDataset(
            sum(data, []), flip_h=flip_h, flip_v=flip_v, flip_hv=flip_hv,
        )

    def train(self, traindata: GamePlayDataset, epochs: int = 5, batch_size: int = 64):
        # TODO: add lr scheduler
        optimizer = optim.Adam(self.model.parameters(), lr=5e-4)
        criterion_p, criterion_v = nn.CrossEntropyLoss(), nn.MSELoss()
        dataloader = DataLoader(traindata, batch_size=batch_size, shuffle=True)
        self.model.train()
        total_loss = running_v_loss = running_p_loss = 0.0
        action_size = self.game.get_action_space()
        for epoch in range(1, epochs + 1):
            for i, data in enumerate(dataloader, 0):
                boards, target_ps, target_vs = data
                # NOTE: Boards are encoded as: (batch_size, nchannel, width, height) to
                # ensure compatibility with conv2d layers
                boards = boards.float().view(
                    -1, 1, self.game.cfg.nrow, self.game.cfg.ncol
                )
                target_ps = target_ps.float().view(-1, action_size)
                target_vs = target_vs.float().view(-1, 1)

                # Predict
                out_ps, out_vs = self.model(boards)
                out_ps, out_vs = out_ps.view(-1, action_size), out_vs.view(-1, 1)

                # Calculate loss
                v_loss: torch.Tensor = criterion_v(out_vs, target_vs)
                p_loss: torch.Tensor = criterion_p(out_ps, target_ps)
                loss = v_loss + p_loss

                # Backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_v_loss += v_loss.item()
                running_p_loss += p_loss.item()
                total_loss += loss.item()
        # Print summary at end of iter
        print(
            f"\t[{epoch:d}, {batch_size:2d}]",
            f"Policy Loss:\t{running_p_loss/(epoch*batch_size)}",
            f"Value Loss:\t{running_v_loss/(epoch*batch_size)}",
            f"Total Loss:\t{total_loss/(epoch*batch_size)}",
            sep="\n\t",
        )

    def save_checkpoint(self, folder, filename):
        if not os.path.exists(folder):
            os.mkdir(folder)
        filepath = os.path.join(folder, filename)
        torch.save({"state_dict": self.model.state_dict()}, filepath)
