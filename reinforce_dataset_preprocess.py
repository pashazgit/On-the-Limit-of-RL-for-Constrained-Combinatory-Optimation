
import sys
# sys.path.insert(0, "./")
import os
import numpy as np
import argparse
import h5py
from tqdm import tqdm
import torch
import torch.nn.functional as F
import pdb


# ----------------------------------------
# Some nice macros to be used for arparse
def str2bool(v):
    return v.lower() in ("true", "1")


# ----------------------------------------
# Global variables within this script
parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", type=str,
                    default="/home/pasha/scratch/datasets/graphs2",
                    help="")

# parser.add_argument("--data_dir", type=str,
#                     default="/home/mostafa/Downloads/datasets",
#                     help="")

parser.add_argument("--n_phones", type=int,
                    default=20,
                    help="number of phones")

parser.add_argument("--n_hosts", type=int,
                    default=5,
                    help="number of hosts")

parser.add_argument("--batch", type=int,
                    default=128,
                    help="")


def main():
    args = parser.parse_args()

    h5_file_path_tr = os.path.join(args.data_dir, 'tr_100.h5')
    h5_tr = h5py.File(h5_file_path_tr, mode='w')
    h5_tr.create_dataset("adj", (args.batch * 100, args.n_phones, args.n_phones), dtype=np.float32)
    h5_tr.create_dataset("x", (args.batch * 100, args.n_phones, args.n_hosts), dtype=np.float32)

    for _i in range(100):

        adj = torch.triu(torch.rand(args.batch, args.n_phones, args.n_phones), diagonal=1)
        adj = adj + adj.transpose(dim0=1, dim1=2)
        adj = adj + torch.eye(args.n_phones)
        # Generate random allocation matrix for val dataset
        hot_idx = torch.randint(high=args.n_hosts, size=(args.batch * args.n_phones,))  # host index
        x = F.one_hot(hot_idx, num_classes=args.n_hosts).reshape(
            args.batch, args.n_phones, args.n_hosts).type(torch.float32)

        idx1 = args.batch * _i
        idx2 = args.batch * (_i + 1)

        h5_tr['adj'][idx1:idx2] = adj.numpy()
        h5_tr['x'][idx1:idx2] = x.numpy()

    h5_tr.close()


if __name__ == "__main__":
    main()
