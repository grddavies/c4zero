import contextlib
import json
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
@click.option('--batch',    help='Batch size', metavar='INT', type=click.IntRange(min=1), required=True)
@click.option('--n_eps',    help='Number of games per training iteration', metavar='INT', type=click.IntRange(min=1), required=True)
@click.option('--n_iters',  help='Number of iterations of training loop', metavar='INT', type=click.IntRange(min=1), required=True)
@click.option('--n_sim',    help='Number of MCTS simulations per move', metavar='INT', type=click.IntRange(min=1), required=True)
# Optional:
@click.option('--eval',     help='Evaluate new model against best model each iteration', metavar='[None|FLOAT]', type=click.FloatRange(min=0.50, max=1.0), default=0.55, show_default=True)
@click.option('--epochs',   help='number of passes of the entire training dataset per iteration', metavar='INT', type=click.IntRange(min=1), default=1, show_default=True)
@click.option('--resume',   help='Resume training from a previous run', metavar='[PATH|STR]',  type=str)
@click.option('--mirror',   help='Reflect game board to augment training data', metavar='[BOOL]', type=bool, default=False, show_default=True)
@click.option('--snap',     help='How frequently to save model checkpoints', metavar='ITERS', type=click.IntRange(min=1), default=1, show_default=True)
def main(**kwargs):
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
    with open(os.path.join(run_dir, 'training_options.json'), 'wt') as f:
        json.dump(opts, f, indent=2)

    c4_cfg = ConnectGameConfig(nrow=6, ncol=7, win_cond=4)
    c4_game = ConnectGame(c4_cfg)
    model = C4Zero()
    trainer = Trainer(c4_game, model, n_jobs=-1)
    for i in range(1, opts.n_iters + 1):
        print(f"[{i}/{opts.n_iters}]")
        print(f"\tExecuting self-play for {opts.n_eps} episodes...")
        training_data = trainer.selfplay(opts.n_eps, opts.n_sim)
        torch.save(training_data, os.path.join(opts.outdir, f"{i:02d}_traindata.pt"))
        print(f"\tTraining for {opts.epochs} epochs...")
        with open('log.txt', 'a') as f:
            with contextlib.redirect_stdout(f):
                trainer.train(training_data, epochs=opts.epochs, batch_size=opts.batch)
                print("\tEvaluating new model...")
        w, l, d = trainer.evaluate(n_games=opts.n_eps, win_frac=opts.eval)
        with open("log.txt", mode="a") as f:
            f.write(f"\nNew model: W:{w:d} L:{l:d} D: {d:d}\n")

        if not i % opts.snap:
            trainer.save_checkpoint(os.path.join(run_dir, 'model_{i:02d}.json'))


if __name__ == "__main__":
    main()
