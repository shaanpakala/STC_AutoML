import pdb
import wandb
import argparse

from dotmap import DotMap
from TuckER import *
from read import *
from utils import *

def parse_args():
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument('--rpath', type=str, default='./')
    parser.add_argument('--dataset', type=str, help='Dataset')
    parser.add_argument('--sparsity', type=float, help='0.1 or 0.5')
    # Model and its hyper parameters
    parser.add_argument('--tf', type=str, help='TF method')
    parser.add_argument('--rank', type=int, help='Rank size of tf', nargs='?')
    parser.add_argument('--wd', type=float, help='Weight decay', nargs='?')
    parser.add_argument('--lr', type=float, help='Learning rate', nargs='?')
    parser.add_argument('--input_dropout', type=float, help='Input dropout')
    parser.add_argument('--hidden_dropout1', type=float, help='Hidden dropout1')
    parser.add_argument('--hidden_dropout2', type=float, help='Hidden dropout2')
    parser.add_argument('--bs', type=int, help='Batch size', nargs='?')
    parser.add_argument('--epochs', type=int, help='Iteration for training', default=100)
    
    # Etc

    args = parser.parse_args()
    dict_args = DotMap(dict(args._get_kwargs()))
    dict_args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return dict_args

def main():

    cfg = parse_args()
    cfg.verbose = True
    # wandb = None
    run = wandb.init(
            project='EXP',
            group=cfg.dataset,
            job_type=str(cfg.rank),
            config = cfg,
    )

    tensor = read_data(cfg, sparsify=cfg.sparsity)
    cfg.sizes = tensor.sizes
    model = train(tensor, cfg, wandb, verbose=cfg.verbose)

    if wandb is not None:
        cfg.wnb_name = wandb.run.name
        wandb.config['model_path'] = save_checkpoints(model, cfg)



if __name__ ==  '__main__':
    main()


