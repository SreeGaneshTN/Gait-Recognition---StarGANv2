import os
import argparse
from munch import Munch
from torch.backends import cudnn
import torch
from dataset import get_train_loader
from dataset import get_test_loader
from dataset import get_val_loader
from solver import Solver


def subdirs(dname):
    return [d for d in os.listdir(dname)
            if os.path.isdir(os.path.join(dname, d))]


def main(args):
    print(args)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    if args.mode == 'train':
        loaders = Munch(src=get_train_loader(root=args.train_img_dir,
                                             img_size=args.img_size,
                                             batch_size=args.batch_size,
                                             num_workers=args.num_workers),
                        val=get_val_loader(root=args.val_img_dir,
                                            img_size=args.img_size,
                                            batch_size=args.val_batch_size,
                                            num_workers=args.num_workers,shuffle=True,))
        solver = Solver(args,loaders)
        solver.Train_network()
    elif args.mode == 'sample':
        loaders = Munch(test=get_test_loader(root=args.src_dir,
                                            img_size=args.img_size,
                                            batch_size=args.val_batch_size,
                                            num_workers=args.num_workers,
                                            shuffle=False))
        solver = Solver(args,loaders)
        solver.sample()
    else:
        raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model arguments
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image resolution')
    parser.add_argument('--num_domains', type=int, default=2,
                        help='Number of domains')
    parser.add_argument('--latent_dim', type=int, default=16,
                        help='Latent vector dimension')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension of mapping network')
    parser.add_argument('--style_dim', type=int, default=64,
                        help='Style code dimension')

    # weight for objective functions
    parser.add_argument('--lambda_reg', type=float, default=1,
                        help='Weight for R1 regularization')
    parser.add_argument('--lambda_cyc', type=float, default=1,
                        help='Weight for cyclic consistency loss')
    parser.add_argument('--lambda_sty', type=float, default=1,
                        help='Weight for style reconstruction loss')
    parser.add_argument('--lambda_ds', type=float, default=1,
                        help='Weight for diversity sensitive loss')
    parser.add_argument('--ds_iter', type=int, default=100000,
                        help='Number of iterations to optimize diversity sensitive loss')
    parser.add_argument('--w_hpf', type=float, default=0,
                        help='weight for high-pass filtering')

    # training arguments
    parser.add_argument('--total_iters', type=int, default=100000,
                        help='Number of total iterations')
    parser.add_argument('--resume_iter', type=int, default=0,
                        help='Iterations to resume training/testing')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=32,
                        help='Batch size for validation')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for Discriminator, Style Encoder and Generator')
    parser.add_argument('--f_lr', type=float, default=1e-6,
                        help='Learning rate for Mapping Network')
    parser.add_argument('--beta1', type=float, default=0.0,
                        help='Decay rate for 1st moment of Adam')
    parser.add_argument('--beta2', type=float, default=0.99,
                        help='Decay rate for 2nd moment of Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--num_outs_per_domain', type=int, default=10,
                        help='Number of generated images per domain during sampling')

    # misc
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'sample', 'eval'],
                        help='This argument is used in solver')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers used in DataLoader')
    parser.add_argument('--seed', type=int, default=777,
                        help='Seed for random number generator')

    # directory for training
    parser.add_argument('--train_img_dir', type=str, default='./GEI/',
                        help='Directory containing training images')
    parser.add_argument('--val_img_dir', type=str, default='./GEI/',
                        help='Directory containing validation images')
    parser.add_argument('--sample_dir', type=str, default='./samples',
                        help='Directory for saving generated images')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory for saving network checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Directory for saving all the Logs')

    # directory for calculating metrics


    # directory for testing
    parser.add_argument('--result_dir', type=str, default='./results',
                        help='Directory for saving generated images ')
    parser.add_argument('--src_dir', type=str, default='/GEI/',
                        help='Directory containing input source images')

    # step size
    parser.add_argument('--print_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=5000)
    parser.add_argument('--save_step', type=int, default=10000)
    parser.add_argument('--eval_step', type=int, default=50000)

    args = parser.parse_args()
    main(args)