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
from scipy.interpolate import spline
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

parser.add_argument("--data_dir", type=str,
                    default="/home/pasha/scratch/jobs/output/RL-CO/dataset",
                    help="Directory to save the best model")

parser.add_argument("--log_dir", type=str,
                    default="/home/pasha/scratch/jobs/output/RL-CO/logs",
                    help="Directory to save logs")


parser.add_argument("--graph", type=int,
                    default=0,
                    help="")

parser.add_argument("--n_phones", type=int,
                    default=20,
                    help="")

parser.add_argument("--q", type=str,
                    default="[2, 2, 2, 2, 2]",
                    help="capacities")

parser.add_argument("--name_idx", type=str,
                    default='0',
                    help="")


def main():
    """The main function."""
    # if not os.path.exists('/home/mostafa/Desktop/New_folder/graphs.pickle'):
    #     handle = open('/home/mostafa/Desktop/New_folder/graphs.pickle', 'wb')
    #     pickle.dump({}, handle)
    # handle = open('/home/mostafa/Desktop/New_folder/graphs.pickle', 'rb')
    # cont = pickle.load(handle)
    # for _i in range(3):
    #     adj = torch.triu(torch.rand(500, 500), diagonal=1)
    #     adj = adj + adj.transpose(dim0=0, dim1=1)
    #     adj = adj + torch.eye(1000)
    #     adj_c = (1 - adj).numpy()
    #     x0 = torch.rand(500 * 5, ).numpy()
    #     cont.update({'adj_c_{}'.format(_i): adj_c, 'x0_{}'.format(_i): x0})
    # handle = open('/home/mostafa/Desktop/New_folder/graphs.pickle', 'wb')
    # pickle.dump(cont, handle)

    args = parser.parse_args()

    # text_path = os.path.join(args.log_dir, "text_{}.txt".format(args.name_idx))
    # text_handle = open(text_path, 'a')

    log_path = os.path.join(args.log_dir, "log_qp_gr_{}.pickle".format(args.name_idx))
    if not os.path.exists(log_path):
        log_handle = open(log_path, 'wb')
        pickle.dump({}, log_handle)
    log_handle = open(log_path, 'rb')
    log_cont = pickle.load(log_handle)

    data_path = os.path.join(args.data_dir, "graphs.pickle")
    data_handle = open(data_path, 'rb')
    data_cont = pickle.load(data_handle)

    n_phones = args.n_phones
    graph = args.graph

    rc = list(map(int, args.q.strip('[]').split(',')))  # relative capacities
    mu = n_phones / 10
    q = np.array([a * mu for a in rc])

    arr = diags([1] * n_phones).toarray()
    M1 = np.concatenate([arr] * 5, axis=1)
    print(M1.shape)
    v1 = np.ones((n_phones, 1))
    print(v1.shape)

    e = np.ones((1, n_phones))
    arr = [e for _ in range(5)]
    M2 = block_diag(arr).toarray()
    print(M2.shape)
    v2 = q.reshape(5, 1)
    print(v2.shape)

    I = -np.eye(n_phones * 5)
    G = np.vstack((M2, I))
    print(G.shape)
    h = np.vstack((q.reshape((5, 1)), np.zeros((n_phones * 5, 1))))
    print(h.shape)

    bnds = tuple([(0, None)] * n_phones * 5)

    cons1 = ({'type': 'eq', 'fun': cons_fun1, 'args': [v1, M1]},
             {'type': 'ineq', 'fun': cons_fun2, 'args': [v2, M2]},
             {'type': 'ineq', 'fun': cons_fun3})

    cons2 = ({'type': 'eq', 'fun': cons_fun1, 'args': [v1, M1]},
             {'type': 'ineq', 'fun': cons_fun2, 'args': [v2, M2]})

    cons3 = ({'type': 'eq', 'fun': cons_fun1, 'args': [v1, M1]},
             {'type': 'ineq', 'fun': cons_fun4, 'args': [h, G]})

    # adj = torch.triu(torch.rand(n_phones, n_phones), diagonal=1)
    # adj = adj + adj.transpose(dim0=0, dim1=1)
    # adj = adj + torch.eye(n_phones)
    # adj_c = (1 - adj).numpy()
    #
    # x0 = torch.rand(n_phones * 5, ).numpy()

    adj_c = data_cont['adj_c_{}'.format(graph)][0:n_phones, 0:n_phones]
    x0 = data_cont['x0_{}'.format(graph)][0:n_phones * 5]

    arr = [adj_c for _ in range(5)]
    A_c = block_diag(arr).toarray()

    # Quadratic programming
    best_risk = np.inf

    res1 = minimize(loss, x0, jac=jac, hess=hess, args=A_c, method='trust-constr', constraints=cons1)
    if res1.success:
        x = res1.x.reshape(5, n_phones)
        c = np.zeros(5, )
        s_idx = x.T.argsort(axis=-1)
        s = np.zeros((n_phones, 5))
        for _l in range(n_phones):
            _j = 4
            while True:
                host = s_idx[_l, _j]
                if c[host] < q[host]:
                    c[host] += 1
                    s[_l, host] = 1
                    break
                else:
                    _j -= 1
        risk = np.dot(s.T.reshape(1, -1), np.dot(A_c, s.T.reshape(-1, 1))).item()
        log_cont.update({'qp_risk_{}_{}_1'.format(n_phones, graph): [True, np.max(x.T, axis=-1), res1.fun, risk]})
        if min(res1.fun, risk) < best_risk:
            best_risk = min(res1.fun, risk)
    else:
        log_cont.update({'qp_risk_{}_{}_1'.format(n_phones, graph): [False, res1.message]})

    res2 = minimize(loss, x0, jac=jac, hess=hess, args=A_c, method='trust-constr', constraints=cons3)
    if res2.success:
        x = res2.x.reshape(5, n_phones)
        c = np.zeros(5, )
        s_idx = x.T.argsort(axis=-1)
        s = np.zeros((n_phones, 5))
        for _l in range(n_phones):
            _j = 4
            while True:
                host = s_idx[_l, _j]
                if c[host] < q[host]:
                    c[host] += 1
                    s[_l, host] = 1
                    break
                else:
                    _j -= 1
        risk = np.dot(s.T.reshape(1, -1), np.dot(A_c, s.T.reshape(-1, 1))).item()
        log_cont.update({'qp_risk_{}_{}_2'.format(n_phones, graph): [True, np.max(x.T, axis=-1), res2.fun, risk]})
        if min(res2.fun, risk) < best_risk:
            best_risk = min(res2.fun, risk)
    else:
        log_cont.update({'qp_risk_{}_{}_2'.format(n_phones, graph): [False, res2.message]})

    if n_phones <= 60:
        res3 = minimize(loss, x0, jac=jac, args=A_c, method='SLSQP', bounds=bnds, constraints=cons2)
        if res3.success:
            x = res3.x.reshape(5, n_phones)
            c = np.zeros(5, )
            s_idx = x.T.argsort(axis=-1)
            s = np.zeros((n_phones, 5))
            for _l in range(n_phones):
                _j = 4
                while True:
                    host = s_idx[_l, _j]
                    if c[host] < q[host]:
                        c[host] += 1
                        s[_l, host] = 1
                        break
                    else:
                        _j -= 1
            risk = np.dot(s.T.reshape(1, -1), np.dot(A_c, s.T.reshape(-1, 1))).item()
            log_cont.update({'qp_risk_{}_{}_3'.format(n_phones, graph): [True, np.max(x.T, axis=-1), res3.fun, risk]})
            if min(res3.fun, risk) < best_risk:
                best_risk = min(res3.fun, risk)
        else:
            log_cont.update({'qp_risk_{}_{}_3'.format(n_phones, graph): [False, res3.message]})

        res4 = minimize(loss, x0, jac=jac, args=A_c, method='SLSQP', constraints=cons1)
        if res4.success:
            x = res4.x.reshape(5, n_phones)
            c = np.zeros(5, )
            s_idx = x.T.argsort(axis=-1)
            s = np.zeros((n_phones, 5))
            for _l in range(n_phones):
                _j = 4
                while True:
                    host = s_idx[_l, _j]
                    if c[host] < q[host]:
                        c[host] += 1
                        s[_l, host] = 1
                        break
                    else:
                        _j -= 1
            risk = np.dot(s.T.reshape(1, -1), np.dot(A_c, s.T.reshape(-1, 1))).item()
            log_cont.update({'qp_risk_{}_{}_4'.format(n_phones, graph): [True, np.max(x.T, axis=-1), res4.fun, risk]})
            if min(res4.fun, risk) < best_risk:
                best_risk = min(res4.fun, risk)
        else:
            log_cont.update({'qp_risk_{}_{}_4'.format(n_phones, graph): [False, res4.message]})

        res5 = minimize(loss, x0, jac=jac, args=A_c, method='SLSQP', constraints=cons3)
        if res5.success:
            x = res5.x.reshape(5, n_phones)
            c = np.zeros(5, )
            s_idx = x.T.argsort(axis=-1)
            s = np.zeros((n_phones, 5))
            for _l in range(n_phones):
                _j = 4
                while True:
                    host = s_idx[_l, _j]
                    if c[host] < q[host]:
                        c[host] += 1
                        s[_l, host] = 1
                        break
                    else:
                        _j -= 1
            risk = np.dot(s.T.reshape(1, -1), np.dot(A_c, s.T.reshape(-1, 1))).item()
            log_cont.update({'qp_risk_{}_{}_5'.format(n_phones, graph): [True, np.max(x.T, axis=-1), res5.fun, risk]})
            if min(res5.fun, risk) < best_risk:
                best_risk = min(res5.fun, risk)
        else:
            log_cont.update({'qp_risk_{}_{}_5'.format(n_phones, graph): [False, res5.message]})

    log_cont.update({'qp_best_risk_{}_{}'.format(n_phones, graph): best_risk})

    # HDF algorithm
    c = np.zeros((5,))
    s = np.zeros((n_phones, 5))
    deg = adj_c.sum(axis=-1).argsort(axis=0)[::-1].reshape(n_phones, )
    for _l in range(n_phones):
        phone = deg[_l]
        corr = adj_c[phone].reshape(n_phones, 1)
        score = np.matmul(s.T, corr).argsort(axis=0).reshape(5, )
        _j = 0
        while True:
            host = score[_j]
            if c[host] < q[host]:
                c[host] += 1
                s[phone, host] = 1
                break
            else:
                _j += 1
    risk = np.trace(np.matmul(s.T, np.matmul(adj_c, s)))
    log_cont.update({'hdf_risk_{}_{}'.format(n_phones, graph): risk})

    # MCF algorithm
    c = np.zeros((5,))
    s = np.zeros((n_phones, 5))
    deg = adj_c.sum(axis=-1).argsort(axis=0).reshape(n_phones, )
    for _l in range(n_phones):
        phone = deg[_l]
        corr = adj_c[phone].reshape(n_phones, 1)
        score = np.matmul(s.T, corr).argsort(axis=0).reshape(5, )
        _j = 0
        while True:
            host = score[_j]
            if c[host] < q[host]:
                c[host] += 1
                s[phone, host] = 1
                break
            else:
                _j += 1
    risk = np.trace(np.matmul(s.T, np.matmul(adj_c, s)))
    log_cont.update({'mcf_risk_{}_{}'.format(n_phones, graph): risk})

    # text_handle.write((res1.x > 1.001).astype(int).sum().astype(str) + ', ' +
    #                (res1.x < -0.001).astype(int).sum().astype(str))

    # text_handle.close()

    log_handle = open(log_path, 'wb')
    pickle.dump(log_cont, log_handle)
    log_handle.close()

    data_handle.close()


def loss(x, A_c):
    return np.dot(x.T, np.dot(A_c, x))


def jac(x, A_c):
    return 2 * np.dot(x.T, A_c)


def hess(x, A_c):
    return 2 * A_c


def cons_fun1(x, v1, M1):
    n_phones = x.size // 5
    x = x.reshape(-1, 1)
    return (v1 - np.dot(M1, x)).reshape(n_phones, )


def cons_fun2(x, v2, M2):
    x = x.reshape(-1, 1)
    return (v2 - np.dot(M2, x)).reshape(5, )


def cons_fun3(x):
    #     x = x.reshape(-1, 1)
    n_phones = x.size // 5
    I = np.eye(n_phones * 5)
    return np.dot(I, x).reshape(n_phones * 5, )


def cons_fun4(x, h, G):
    n_phones = x.size // 5
    x = x.reshape(-1, 1)
    return (h - np.dot(G, x)).reshape((5 + n_phones * 5), )


if __name__ == "__main__":
    main()

