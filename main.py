from game.connect.connect import ConnectGame, ConnectGameConfig
from c4zero import C4Zero, Connect2Model
from trainer import Trainer
import torch

# Set up game of Connect 2
nrow = 6
ncol = 7
win_cond = 4
game_cfg = ConnectGameConfig(nrow, ncol, win_cond)

game = ConnectGame(game_cfg)
last_model = torch.load("latestC4.pt")
model = C4Zero(1, nrow, ncol, game.get_action_space())
model.load_state_dict(last_model["state_dict"])
# model = Connect2Model(nrow, ncol, ncol, "cpu")

trainer = Trainer(game, model, n_jobs=-1)

trainer.learn(n_iters=30, n_eps=100, n_sim=150).save_checkpoint(".", "latestC4.pt")
