from game.connect.connect import ConnectGame, ConnectGameConfig
from c4zero import C4Zero, Connect2Model
from trainer import Trainer


args = {
    "batch_size": 64,
    "n_simulations": 100,  # Number of Monte Carlo simulations for each move
    # "numIters": 500,  # Total number of training iterations
    # "numEps": 100,  # Number of full games (episodes) to run during each iteration
    "numItersForTrainExamplesHistory": 20,
    "epochs": 2,  # Number of epochs of training per iteration
    "checkpoint_path": "latest.pt",  # location to save latest set of weights
}

# Set up game of Connect 2
nrow = 1
ncol = 4
win_cond = 2
game_cfg = ConnectGameConfig(nrow, ncol, win_cond)

game = ConnectGame(game_cfg)
model = Connect2Model(ncol, ncol, "cpu")

trainer = Trainer(game, model, args)
trainer.learn(n_iters=500, n_eps=100).save_checkpoint(".", "latest.pt")
