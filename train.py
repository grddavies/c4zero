import glob
import json
import logging
import os
import re

import click
import torch

import util
from c4zero import C4Zero
from game.connect import ConnectGame, ConnectGameConfig
from trainer import Trainer


@click.command()
# Required:
@click.option('--outdir',   help='Where to save the results', metavar='DIR', required=True)
@click.option('--n_iters',  help='Number of iterations of training loop', metavar='INT', type=click.IntRange(min=1), required=True)
@click.option('--n_eps',    help='Number of games per training iteration', metavar='INT', type=click.IntRange(min=1), required=True)
@click.option('--batch',    help='Batch size', metavar='INT', type=click.IntRange(min=1), required=True)
@click.option('--n_sim',    help='Number of MCTS simulations per move', metavar='INT', type=click.IntRange(min=1), required=True)
# Optional:
@click.option('--eval',     help='Proportion of wins new model must achieve to be kept. If zero, evaluation is skipped', metavar='[None|FLOAT]', type=click.FloatRange(max=1.0), default=0.55, show_default=True)
@click.option('--epochs',   help='number of passes of the entire training dataset per iteration', metavar='INT', type=click.IntRange(min=1), default=1, show_default=True)
@click.option('--resume',   help='Resume training from a previous run', metavar='[PATH|STR]',  type=str)
@click.option('--mirror',   help='Reflect game board to augment training data', metavar='[BOOL]', type=bool, default=False, show_default=True)
@click.option('--snap',     help='How frequently to save model checkpoints', metavar='ITERS', type=click.IntRange(min=1), default=1, show_default=True)
def main(**kwargs):
    """
    Iteratively generate data through self-play, learn and evaluate new models

    Example:
    -------

    `python train.py --outdir=training --n_iters=10 --batch=256 --n_eps=150 --n_sim=200`

    Train for 10 iterations, using a batch size of 256 samples with 150 episodes of self-play and 200 MCTS simulations per-move

    """
    # Collect options
    opts = util.EasyDict(kwargs)

    # Make output directory
    outdir = opts.outdir
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    run_dir = os.path.join(outdir, f'{cur_run_id:05d}')
    assert not os.path.exists(run_dir)
    os.makedirs(run_dir)

    # Set up logging
    log_path = os.path.join(run_dir, "log.txt")
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path)
        ]
    )
    logger = logging.getLogger()

    with open(os.path.join(run_dir, 'training_options.json'), 'wt') as f:
        json.dump(opts, f, indent=2)

    # Set up game
    c4_cfg = ConnectGameConfig(nrow=6, ncol=7, win_cond=4)
    c4_game = ConnectGame(c4_cfg)
    trainer = Trainer(c4_game, C4Zero(), n_jobs=-1)
    logger.info(f"training on {trainer.device} for {opts.n_iters} iters")

    # TODO: Allow loading progress from saved
    if opts.resume is not None:
        if opts.resume == "latest":
            models = glob.glob(os.path.join(outdir, prev_run_dirs[-1], "model_*.pt"))
            logger.info(f"Resuming from {models[-1]}")
            trainer.load_checkpoint(models[-1])
        else:
            raise ValueError(f"Unknown option for --resume: {opts.resume}")

    # Training loop
    for i in range(1, opts.n_iters + 1):
        logger.info(f"[{i:02d}/{(opts.n_iters):02d}]")
        logger.info(f"Executing self-play for {opts.n_eps} episodes...")
        training_data = trainer.selfplay(opts.n_eps, opts.n_sim)
        torch.save(training_data, os.path.join(run_dir, f"{i:03d}_traindata.pt"))
        logger.info(f"Training for {opts.epochs} epochs...")
        trainer.train(training_data, epochs=opts.epochs, batch_size=opts.batch)
        logger.debug(f"Trained on {len(training_data) * opts.epochs} samples")
        new_model = True
        if opts.eval:
            logger.info("Evaluating new model...")
            new_model = trainer.evaluate(n_games=opts.n_eps, win_frac=opts.eval)

        if (not i % opts.snap) and new_model:
            logger.info("Saving new model")
            trainer.save_checkpoint(os.path.join(run_dir, f'model_{i:03d}.pt'))


if __name__ == "__main__":
    main()
