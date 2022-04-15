# pyright: reportUnknownMemberType=false, reportMissingTypeStubs=false
import logging
from typing import Any, Dict, List, Tuple, Optional

import joblib
import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

from arena import Arena
from c4zero import C4Zero
from game.base import Game, GamePlayDataset, GameState
from game.connect.players import AI
from MCTS import MCTS
from util import ProgressParallel

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, game: Game, model: C4Zero, n_jobs: Optional[int] = None):
        """
        Parameters
        ----------

        n_jobs: Optional int (default = None)
            Number of CPUs to use during self play. `None` means 1 unless in a
            :obj:`joblib.parallel_backend` context. `-1` means using all processors.
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.model.to(self.device)
        self.best_model = self.model.clone()
        self.best_model.to(self.device)
        self.game = game
        self.mcts = MCTS(game, self.model)
        self.n_jobs: int = joblib.parallel.effective_n_jobs(n_jobs)  # type: ignore

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
            action = root.game.get_action(action_index)
            # Update the starting state in MCTS
            self.mcts.game = self.mcts.game.move(action)
            n_moves += 1

            if not self.mcts.game.over:
                continue
            # Reward from the perspective of the current player at game end
            reward = self.mcts.game.reward_player(self.mcts.game.state.current_player)
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
        self.best_model = self.model.clone()
        for i in range(1, n_iters + 1):
            print(f"[{i}/{n_iters}]")
            print(f"\tExecuting self-play for {n_eps} episodes...")
            traindata = self.selfplay(n_eps, n_sim)
            print("\tTraining...")
            self.train(traindata, epochs=epochs, batch_size=batch_size)
            print("Evaluating...")
            self.evaluate(n_eps // 2)
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
        Run games of self play and return output as a dataset

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
        data: List[List[Tuple[np.ndarray, np.ndarray, float]]] = ProgressParallel(
            self.n_jobs, total=n_eps, leave=False  # type: ignore
        )(
            joblib.delayed(self.execute_episode)(n_sim, t_mcts, t_move_thresh)
            for _ in range(n_eps)
        )
        return GamePlayDataset(
            sum(data, []), flip_h=flip_h, flip_v=flip_v, flip_hv=flip_hv,
        )

    def train(self, traindata: GamePlayDataset, epochs: int = 5, batch_size: int = 64):
        optimizer = optim.Adam(self.model.parameters(), lr=5e-4, weight_decay=1e-4)

        # # Settings recreate the 2017 paper:
        # optimizer = optim.SGD(
        #     self.model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4
        # )
        # lr_scheduler = optim.lr_scheduler.MultiStepLR(
        #     optimizer, milestones=[400, 600, 800], gamma=0.1
        # )
        criterion_p, criterion_v = nn.CrossEntropyLoss(), nn.MSELoss()
        dataloader = DataLoader(traindata, batch_size=batch_size, shuffle=True)
        self.model.train()
        total_loss = running_v_loss = running_p_loss = 0.0
        action_size = self.game.get_action_space()
        for _ in range(1, epochs + 1):  # iter epochs
            for _, minibatch in enumerate(dataloader):
                boards, target_ps, target_vs = minibatch
                # NOTE: Boards are encoded as: (batch_size, nchannel, width, height) to
                # ensure compatibility with conv2d layers
                boards = (
                    boards.float()
                    .view(-1, 1, self.game.cfg.nrow, self.game.cfg.ncol)
                    .to(self.device)
                )
                target_ps = target_ps.float().view(-1, action_size).to(self.device)
                target_vs = target_vs.float().view(-1, 1).to(self.device)

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
                # lr_scheduler.step()

                running_v_loss += v_loss.item()
                running_p_loss += p_loss.item()
                total_loss += loss.item()
        # Print summary at end of iter
        logger.info(f"[{(1+epochs):d}, {batch_size}]")
        logger.info(f"Policy Loss:\t{running_p_loss/((1+epochs)*batch_size)}")
        logger.info(f"Value Loss:\t{running_v_loss/((1+epochs)*batch_size)}")
        logger.info(f"Total Loss:\t{total_loss/((1+epochs)*batch_size)}")
        return self

    def evaluate(self, n_games: int = 400, win_frac: float = 0.55):
        """Compare current model agains best model"""
        p1 = AI(self.model, n_sim=150)
        p2 = AI(self.best_model, n_sim=150)
        arena = Arena(p1, p2, self.game, self.n_jobs)
        w, l, d = arena.play_games(n_games)
        logger.info(f"New model: W:{w:d} L:{l:d} D: {d:d}")
        if w / sum((w, l, d)) > win_frac:
            self.best_model = self.model.clone()
            return True
        # If model wins =< win_frac, continue training from best model
        self.model = self.best_model.clone()
        return False

    def save_checkpoint(self, filename: str):
        torch.save({"state_dict": self.model.state_dict()}, filename)

    def load_checkpoint(self, filepath: str):
        checkpoint: Dict[str, Any] = torch.load(filepath)
        self.model.load_state_dict(checkpoint["state_dict"])
