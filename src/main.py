import argparse
import logging
import sys
from datetime import date

import torch
from torch import nn
from torch.optim import Adam, AdamW

from trainers.mlp_trainer import (
    ESSFashionMNISTSimplexSubspaceMLPTrainer, FashionMNISTMLPTrainer,
    FashionMNISTSimplexSubspaceMLPTrainer, FashionMNISTSubspaceMLPTrainer,
    MFVIFashionMNISTSimplexSubspaceMLPTrainer, EnsembleFashionMNISTMLPTrainer,
    SubspaceSamplingFashionMNISTSimplexSubspaceMLPTrainer)

arg_trainer_map = {
    'f_mnist_mlp':
    FashionMNISTMLPTrainer,
    'f_mnist_subspace_mlp':
    FashionMNISTSubspaceMLPTrainer,
    'f_mnist_simplex_subspace_mlp':
    FashionMNISTSimplexSubspaceMLPTrainer,
    'mfvi_f_mnist_simplex_subspace_mlp':
    MFVIFashionMNISTSimplexSubspaceMLPTrainer,
    'ess_f_mnist_simplex_subspace_mlp':
    ESSFashionMNISTSimplexSubspaceMLPTrainer,
    'ensemble_f_mnist_mlp':
    EnsembleFashionMNISTMLPTrainer,
    'subspace_sampling_f_mnist_simplex_subspace_mlp':
    SubspaceSamplingFashionMNISTSimplexSubspaceMLPTrainer,
}
arg_optimizer_map = {'adamW': AdamW, 'adam': Adam}


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Experiments with neural loss subspace inference')

    parser.add_argument('--epochs',
                        default=50,
                        type=int,
                        help='number of epochs to train model')
    parser.add_argument('--device',
                        '-d',
                        default='cuda',
                        type=str,
                        help='cpu or gpu ID to use')
    parser.add_argument('--batch_size',
                        default=128,
                        type=int,
                        help='mini-batch size used to train model')
    parser.add_argument('--dropout_prob',
                        default=0.3,
                        type=float,
                        help='probability for dropout layers')
    parser.add_argument('--save_dir', help='path to saved model files')
    parser.add_argument('--data_dir', help='path to data files')
    parser.add_argument('--optimizer',
                        default='adamW',
                        help='type of optimizer to use')
    parser.add_argument('--num_repeats',
                        default=3,
                        type=int,
                        help='number of times to repeat experiment')
    parser.add_argument('--seed',
                        default=11202022,
                        type=int,
                        help='random seed to be used in numpy and torch')
    parser.add_argument('--learning_rate',
                        default=1e-3,
                        type=float,
                        help='learning rate for optimizer')
    parser.add_argument('--hidden_size',
                        default=512,
                        type=int,
                        help='dimensionality of hidden layers')
    parser.add_argument('--trainer_type',
                        default='f_mnist_mlp',
                        help='type of experiment to run')
    parser.add_argument(
        '--val_midpoint_only',
        action='store_true',
        help=
        'only collect validation metrics for the midpoint of the line (for speed)'
    )
    parser.add_argument('--beta',
                        default=1.0,
                        type=float,
                        help='constant for learning subspaces')
    parser.add_argument(
        '--infer_posterior_only',
        action='store_true',
        help='just do posterior inference for saved trained model')
    parser.add_argument('--eval_ensemble_test',
                        action='store_true',
                        help='evaluate posterior/ensemble on test set')
    parser.add_argument('--eval_ensemble_val',
                        action='store_true',
                        help='evaluate posterior/ensemble on validation set')

    args = parser.parse_args()
    configs = args.__dict__

    # for repeatability
    torch.manual_seed(configs['seed'])

    # set up logging
    filename = f'{configs["trainer_type"]}-{date.today()}'
    FORMAT = '%(asctime)s;%(levelname)s;%(message)s'
    logging.basicConfig(level=logging.INFO,
                        filename=f'{configs["save_dir"]}logs/{filename}.log',
                        filemode='a',
                        format=FORMAT)
    logging.info(configs)

    # get trainer
    trainer_type = arg_trainer_map[configs['trainer_type']]
    trainer = trainer_type(
        optimizer_type=arg_optimizer_map[configs['optimizer']],
        criterion=nn.CrossEntropyLoss(reduction='sum'),
        **configs)

    if configs['eval_ensemble_test']:
        trainer.create_testloader()
        trainer.eval_posterior(trainer.test_loader)
    elif configs['eval_ensemble_val']:
        trainer.create_dataloaders()
        trainer.eval_posterior(trainer.valid_loader)
    else:
        # perform experiment n times
        for iter in range(configs['num_repeats']):
            if configs['infer_posterior_only']:
                trainer.create_testloader()
                trainer.create_dataloaders()
                trainer.load_model(iter)
                trainer.fit_and_eval_posterior()
            else:
                trainer.run_experiment(iter)

    return 0


if __name__ == '__main__':
    sys.exit(main())
