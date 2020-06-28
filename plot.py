import sys
# sys.path.insert(0, "./")
import os
import math
import random
import numpy as np
from collections import namedtuple
from itertools import count
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image
from tqdm import tqdm
import h5py
import argparse
from scipy import optimize
from scipy.optimize import minimize
from scipy.sparse import coo_matrix, block_diag, diags
# from scipy.interpolate import spline
import pickle
import queue
import pdb

import torch
from torch import nn, optim
from tensorboardX import SummaryWriter
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.nn import init


# ----------------------------------------
# Some nice macros to be used for arparse
def str2bool(v):
    return v.lower() in ("true", "1")


# -----------------------------------------
# Global variables within this script
parser = argparse.ArgumentParser()

parser.add_argument("--log_dir", type=str,
                    default="/home/mostafa/Uvic/Thesis/implementation/RL-CO/logs/logs",
                    help="Directory to save logs")

parser.add_argument("--name_idx", type=str,
                    default='11233',
                    help="")


def main():

    args = parser.parse_args()

    n_phones_list = [20, 30, 40, 50, 60, 120, 200]
    test_batch = 1

    risk_qp_list = []
    risk_hdf_list = []
    risk_mcf_list = []

    log_path = os.path.join(args.log_dir, "log_qp_gr_{}.pickle".format(args.name_idx))
    if not os.path.exists(log_path):
        log_handle = open(log_path, 'wb')
        pickle.dump({}, log_handle)
    log_handle = open(log_path, 'rb')
    log_cont = pickle.load(log_handle)

    for n_phones in n_phones_list:
        risk_list1 = []
        risk_list2 = []
        risk_list3 = []
        for graph in range(test_batch):
            risk_list1.append(log_cont['qp_best_risk_{}_{}'.format(n_phones, graph)])
            risk_list2.append(log_cont['hdf_risk_{}_{}'.format(n_phones, graph)])
            risk_list3.append(log_cont['mcf_risk_{}_{}'.format(n_phones, graph)])
        a1 = np.asarray(risk_list1)
        a2 = np.asarray(risk_list2)
        a3 = np.asarray(risk_list3)
        a1 = a1[~np.isinf(a1)]
        a2 = a2[~np.isinf(a2)]
        a3 = a3[~np.isinf(a3)]
        print('risk_qp_{}: '.format(n_phones), a1.mean())
        risk_qp_list.append(a1.mean())
        print('risk_hdf_{}: '.format(n_phones), a2.mean())
        risk_hdf_list.append(a2.mean())
        print('risk_mcf_{}: '.format(n_phones), a3.mean())
        risk_mcf_list.append(a3.mean())

    mpl.use('Agg')
    fig = plt.figure()
    # plt.plot(n_phones_list, risk_rl_list, "g--", label="RL")
    plt.plot(n_phones_list, risk_qp_list, "-r", label="QP")
    plt.plot(n_phones_list, risk_hdf_list, "-b", label="HDF")
    plt.plot(n_phones_list, risk_mcf_list, "-g", label="MCF")
    plt.ylabel('potential risk')
    plt.xlabel('n_phones')
    plt.legend(loc='best')
    fig.savefig(os.path.join(args.log_dir, 'plot_qp_gr_{}.png'.format(args.name_idx)))

    log_handle.close()


if __name__ == "__main__":
    main()

