
# sys.path.insert(0, "./adaptive_quantization/input_pipeline")
# sys.path.insert(0, "./adaptive_quantization/utils")
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
                    default="/home/pasha/scratch/datasets/graphs",
                    help="")

parser.add_argument("--phones", type=int,
                    default=200,
                    help="number of phones")

parser.add_argument("--hosts", type=int,
                    default=5,
                    help="number of hosts")

parser.add_argument("--batch", type=int,
                    default=512,
                    help="")

parser.add_argument("--r", type=int,
                    default=0.2,
                    help="r-random graph")


def main():
    args = parser.parse_args()
    np.random.seed(100)
    create_tr_dataset(args)
    create_val_dataset(args)


def create_tr_dataset(args):
    h5_file_path_tr = os.path.join(args.data_dir, 'tr.h5')
    h5_tr = h5py.File(h5_file_path_tr, mode='w')
    h5_tr.create_dataset("adj", (args.batch*1000, args.phones, args.phones), dtype=np.float32)
    h5_tr.create_dataset("al", (args.batch*1000, args.phones, args.hosts), np.float32)

    x = args.batch * args.phones
    for _i in tqdm(range(1000)):
        idx1 = args.batch * _i
        idx2 = args.batch * (_i+1)

        adj = np.random.rand(args.batch, args.phones, args.phones)
        adj = (adj + adj.transpose(0, 2, 1)) / 2
        adj = (adj < args.r).astype('f')
        for _j in range(args.phones):
            adj[:, _j, _j] = 1
        h5_tr['adj'][idx1:idx2] = adj

        # al = np.zeros((x, args.hosts))  # initial allocation
        # h_idx = np.random.randint(args.hosts, size=x)  # host index
        # al[np.arange(len(h_idx)), h_idx] = 1
        h_idx = torch.randint(high=args.hosts, size=(x, ))  # host index
        al = F.one_hot(h_idx, num_classes=args.hosts)
        al = al.reshape(args.batch, args.phones, args.hosts).numpy().astype('f')
        h5_tr['al'][idx1:idx2] = al

    h5_tr.close()


def create_val_dataset(args):
    h5_file_path_val = os.path.join(args.data_dir, 'val.h5')
    h5_val = h5py.File(h5_file_path_val, mode='w')
    h5_val.create_dataset("adj", (10000, args.phones, args.phones), dtype=np.float32)
    h5_val.create_dataset("al", (10000, args.phones, args.hosts), dtype=np.float32)

    adj = np.random.rand(10000, args.phones, args.phones)
    adj = (adj + adj.transpose(0, 2, 1)) / 2
    adj = (adj < args.r).astype('f')
    for _j in range(args.phones):
        adj[:, _j, _j] = 1
    h5_val['adj'][:] = adj

    x = 10000 * args.phones
    # al = np.zeros((x, args.hosts))  # initial allocation
    # h_idx = np.random.randint(args.hosts, size=x)  # host index
    # al[np.arange(len(h_idx)), h_idx] = 1
    h_idx = torch.randint(high=args.hosts, size=(x, ))  # host index
    al = F.one_hot(h_idx, num_classes=args.hosts)
    al = al.reshape(10000, args.phones, args.hosts).numpy().astype('f')
    h5_val['al'][:] = al

    h5_val.close()


if __name__ == "__main__":
    main()
