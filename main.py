from game.connect.connect import ConnectGame, ConnectGameConfig
from c4zero import C4Zero, Connect2Model
from trainer import Trainer

# Set up game of Connect 2
nrow = 6
ncol = 7
win_cond = 4
game_cfg = ConnectGameConfig(nrow, ncol, win_cond)

game = ConnectGame(game_cfg)
model = C4Zero(1, nrow, ncol, game.get_action_space())

trainer = Trainer(game, model)
trainer.learn(n_iters=500, n_eps=100).save_checkpoint(".", "latest.pt")
