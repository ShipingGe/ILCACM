import os
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser("Weakly-supervised Dense Video Caption", add_help=False)

    parser.add_argument('--dataset_name', default='ActivityNet-1.3', choices=['ActivityNet-1.3'])

    parser.add_argument(
        '--train_caption_path',
        default='../datasets/activitynet-1.3/captions/train.json',
        type=str,
        help='Path to the train.json file.'
    )

    parser.add_argument(
        '--val_caption_path',
        type=str, nargs='+',
        default=['../datasets/activitynet-1.3/captions/val_1.json', '../datasets/activitynet-1.3/captions/val_2.json'],
        help='Path to the val.json file.'
    )
    parser.add_argument(
        '--val_caption_1',
        type=str,
        default='../datasets/activitynet-1.3/captions/val_1.json',
        help='Path to the val_1.json file.'
    )

    parser.add_argument(
        '--val_caption_dense_pred',
        default='./outputs/ActivityNet-1.3pred_dense_captions.json',
        type=str,
        help='Path to the pred_dense_captions.json file.'
    )

    parser.add_argument(
        '--test_caption_path',
        default='',
        type=str,
        help='Path to the test.json file, which is not available in the ActivityNet-1.3 dataset :<'
    )

    parser.add_argument(
        '--video_feats_dir',
        default='../datasets/activitynet-1.3/videos/clip-feats',
        type=str,
        help='Path to the feature file.'
    )

    parser.add_argument('--train_batch_size', default=8, type=int)

    parser.add_argument('--eval_batch_size', default=8, type=int)

    parser.add_argument(
        '--base_model_path',
        default='?/models--distilbert--distilgpt2/snapshots/2290a62682d06624634c1f46a6ad5be0f47f38aa/',
        type=str,
        help='Path to the pretrained GPT-2 Model.')

    parser.add_argument('--max_frames', default=32, type=int)

    parser.add_argument('--max_tokens', default=1024, type=int, help='Max number of the tokens for both input and output.')

    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")

    parser.add_argument("--total-epochs", default=20, type=int, help="number of training epochs")

    parser.add_argument("--save_dir",
                        default="./outputs",
                        help="path where to save, empty for no saving")

    parser.add_argument("--device", default="cuda", help="device to use")

    parser.add_argument("--seed", default=42, type=int, help="random seed")

    parser.add_argument("--load",
                        default="",
                        help="path to load checkpoint")

    parser.add_argument("--resume", action="store_true", help="continue training if loading checkpoint")

    parser.add_argument("--print_freq", type=int, default=100, help="print log every print_freq iterations")

    parser.add_argument("--cap-epochs", default=10, type=int, metavar="N", help="start epoch")

    parser.add_argument("--eval", action="store_true", help="only run evaluation")

    parser.add_argument("--num_workers", default=8, type=int, help="number of workers for dataloader")

    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")

    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")

    parser.add_argument("--tau", default=2., type=float, help="tau factor used in gaussian mask learning.")
    parser.add_argument("--gamma", default=0.8, type=float, help="gamma factor used in diversity loss.")
    parser.add_argument("--max_captions", default=12, type=int,
                        help="max number of captions of each video used for training.")
    parser.add_argument("--name", default='name', type=str, help="name for this run.")

    return parser

