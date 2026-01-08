import argparse
import random
import sys

import numpy as np
import torch

from mpcompress.models.fcvq import Dinov2FCVQCodec


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Segmentation eval (FCVQ)")

    # Keep the same argument names as test_bits_seg.py
    parser.add_argument('--data_path', type=str, required=False, default='', help='dataset path (unused)')
    parser.add_argument('--pretrain', type=bool, required=False, default=True, help='unused')
    parser.add_argument('--lr', default=1e-4, type=float, help='unused')
    parser.add_argument('--momentum', default=0.9, type=float, help='unused')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='unused')
    parser.add_argument('-warm_up', default=False, type=bool, help='unused')

    parser.add_argument('-n', '--num-workers', type=int, default=4, help='unused')
    parser.add_argument('--batch_size', type=int, default=16, help='unused')
    parser.add_argument('--test_batch_size', type=int, default=1, help='unused')

    parser.add_argument('--cuda', action='store_true', help='Use cuda')
    parser.add_argument('--save', action='store_true', default=True, help='unused')
    parser.add_argument('--seed', type=int, default=0, help='Set random seed')
    parser.add_argument('--clip_max_norm', default=1.0, type=float, help='unused')
    parser.add_argument('--checkpoint', type=str, default='', help='unused')

    parser.add_argument('--embedding_dim', type=int, default=16)
    parser.add_argument('--num_embeddings', type=int, default=128)
    parser.add_argument('--vq_path', type=str, default="/data/qiaoxichen/model/UniVQ/epoch_100num_128dim_16chunk_1.pth.tar")
    parser.add_argument('--num_chunks', type=int, default=1)
    parser.add_argument('--lmbda', type=float, default=1.0)

    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)

    # reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'

    codec = Dinov2FCVQCodec(
        fcvq_kwargs=dict(
            num_embeddings=args.num_embeddings,
            embedding_dim=args.embedding_dim,
            num_chunks=args.num_chunks,
            lmbda=args.lmbda,
        ),
        build_dino=False,
        freeze_dino=True,
    ).to(device)

    metrics = codec.seg_eval_from_feature_files(
        vq_path=args.vq_path,
        device=device,
    )

    print(f"\t=======Ori-mIOU======: {metrics['miou_ori']}")
    print(f"\t=======Ori-MSE======: {metrics['mse_ori']}")
    print(f"\t=======Recon-mIOU======: {metrics['miou_recon']}")
    print(f"\t=======Recon-MSE======: {metrics['mse_recon']}")
    print(f"Average Data Size (in bpp): {metrics['bpp_avg']}")


if __name__ == '__main__':
    main(sys.argv[1:])
