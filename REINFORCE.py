
import sys
# sys.path.insert(0, "./")
import os
import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm
from tensorboardX import SummaryWriter
import torchvision
import argparse
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import h5py
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
from scipy.sparse import coo_matrix, block_diag, diags
from scipy.optimize import minimize
from torch.nn import init
import math
import pdb


# ----------------------------------------
# Some nice macros to be used for arparse
def str2bool(v):
    return v.lower() in ("true", "1")


# -----------------------------------------
# Global variables within this script
parser = argparse.ArgumentParser()

parser.add_argument("--mode", type=str,
                    default="train",
                    choices=["train", "test"],
                    help="Run mode")

parser.add_argument("--data_dir", type=str,
                    default="/home/pasha/scratch/datasets",
                    help="")

parser.add_argument("--optim", type=str,
                    default="adam",
                    choices=["adam", "rmsprop"],
                    help="")

parser.add_argument("--loss_type", type=str,
                    default="huber",
                    choices=["huber", "square"],
                    help="")

parser.add_argument("--save_dir", type=str,
                    default="/home/pasha/scratch/jobs/output/RL-CO/saves",
                    help="Directory to save the best model")

parser.add_argument("--log_dir", type=str,
                    default="/home/pasha/scratch/jobs/output/RL-CO/logs",
                    help="Directory to save logs")

parser.add_argument("--lr", type=float,
                    default=1e-4,
                    help="Learning rate (gradient step size)")

parser.add_argument("--gamma", type=float,
                    default=0.999,
                    help="")

parser.add_argument("--eps_start", type=float,
                    default=0.9,
                    help="")

parser.add_argument("--eps_end", type=float,
                    default=0.05,
                    help="")

parser.add_argument("--eps_decay", type=float,
                    default=200,
                    help="")

parser.add_argument("--scale", type=float,
                    default=1,
                    help="")

parser.add_argument("--reg", type=float,
                    default=100,
                    help="")

parser.add_argument("--clip", type=str2bool,
                    default=True,
                    help="")

parser.add_argument("--clamp", type=str2bool,
                    default=True,
                    help="")

parser.add_argument("--constrained", type=str2bool,
                    default=True,
                    help="")

parser.add_argument("--target_update", type=int,
                    default=1,
                    help="")

parser.add_argument("--delay_trigger", type=int,
                    default=170,
                    help="")

# parser.add_argument("--init_stddev", type=float,
#                     default=1e-3,
#                     help="Learning rate (gradient step size)")

parser.add_argument("--replay_batch", type=int,
                    default=128,
                    help="")

parser.add_argument("--val_batch", type=int,
                    default=64,
                    help="")

parser.add_argument("--test_batch", type=int,
                    default=32,
                    help="")

parser.add_argument("--n_episodes", type=int,
                    default=50,
                    help="")

parser.add_argument("--embed_dim", type=int,
                    default=64,
                    help="")

parser.add_argument("--n_heads", type=int,
                    default=4,
                    help="")

parser.add_argument("--enc_ff_hidden", type=int,
                    default=256,
                    help="Encoder feed_forward hidden units")

parser.add_argument("--dec_ff_hidden", type=int,
                    default=256,
                    help="Decoder feed_forward hidden units")

parser.add_argument("--n_layers", type=int,
                    default=2,
                    help="")

parser.add_argument("--n_phones", type=int,
                    default=57,
                    help="")

parser.add_argument("--n_hosts", type=int,
                    default=5,
                    help="")

parser.add_argument("--delay", type=int,
                    default=8,
                    help="")

parser.add_argument("--resume", type=str2bool,
                    default=False,
                    help="")

parser.add_argument("--batch", type=int,
                    default=128,
                    help="")

parser.add_argument("--epochs", type=int,
                    default=40,
                    help="")

parser.add_argument("--beta", type=float,
                    default=0.5,
                    help="")

parser.add_argument("--baseline", type=str2bool,
                    default=True,
                    help="")

parser.add_argument("--name_idx", type=str,
                    default='0',
                    help="")


def main():
    args = parser.parse_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test(args)
    else:
        raise ValueError("Unknown run mode \"{}\"".format(args.mode))


def train(args):
    # CUDA_LAUNCH_BLOCKING = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize datasets for both training and validation
    tr_file_path = os.path.join(args.data_dir, 'tr.h5')
    tr_data = graphs(tr_file_path)

    # Create data loader for training and validation.
    tr_loader = DataLoader(
        dataset=tr_data,
        batch_size=args.batch,
        num_workers=2,
        shuffle=True)

    # Create model instance
    model = ModelClass(args).to(device)
    print('\nmodel created')
    model.train()

    # Create optimizier
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # No need to move the optimizer (as of PyTorch 1.0), it lies in the same space as the model

    # Create log directory and save directory if it does not exist
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Create summary writer
    tr_writer = SummaryWriter(
        log_dir=os.path.join(args.log_dir, "train_{}".format(args.name_idx)))
    val_writer = SummaryWriter(
        log_dir=os.path.join(args.log_dir, "valid_{}".format(args.name_idx)))

    # Initialize training
    start_epoch = 0
    steps_done = 0  # make counter start at zero
    best_val_ret = 0  # to check if best validation accuracy
    # Prepare checkpoint file and model file to save and load from
    checkpoint_file = os.path.join(args.save_dir, "checkpoint_{}.path".format(args.name_idx))

    ret_list = []
    val_ret_list = []

    log_path = os.path.join(args.log_dir, "log_reinforce_tr_{}.pickle".format(args.name_idx))
    log_handle = open(log_path, 'wb')
    pickle.dump({}, log_handle)
    log_handle = open(log_path, 'rb')
    log_cont = pickle.load(log_handle)

    val_adj = torch.triu(torch.rand(args.val_batch, args.n_phones, args.n_phones), diagonal=1)
    val_adj = val_adj + val_adj.transpose(dim0=1, dim1=2)
    val_adj = val_adj + torch.eye(args.n_phones)
    val_adj_c = (1 - val_adj).to(device)
    # Generate random allocation matrix for val dataset
    val_hot_idx = torch.randint(high=args.n_hosts, size=(args.val_batch * args.n_phones,))  # host index
    val_x = F.one_hot(val_hot_idx, num_classes=args.n_hosts).reshape(
        args.val_batch, args.n_phones, args.n_hosts).type(torch.float32).to(device)

    risk_init = torch.matmul(torch.matmul(val_x.transpose(dim0=1, dim1=2),
                                          val_adj_c), val_x)
    risk_init = risk_init.diagonal(dim1=1, dim2=2).sum(dim=-1).mean()

    # # Check for existing training results. If it exists, and the configuration
    # # is set to resume `args.resume==True`, resume from previous training. If
    # # not, delete existing checkpoint.
    # if os.path.exists(checkpoint_file):
    #     if args.resume:
    #         print("Checkpoint found! Resuming")
    #         # Read checkpoint file.
    #         load_res = torch.load(
    #             checkpoint_file,
    #             map_location="cpu")
    #         start_epoch = load_res["epoch"]
    #         # Resume iterations
    #         iter_idx = load_res["iter_idx"]
    #         # Resume best va result
    #         best_val_ret = load_res["best_val_ret"]
    #         # Resume model
    #         model.load_state_dict(load_res["model"])
    #         # Resume optimizer
    #         optimizer.load_state_dict(load_res["optimizer"])
    #     else:
    #         os.remove(checkpoint_file)

    # Training loop
    for epoch in tqdm(range(start_epoch, args.epochs)):
        for adj, x in tr_loader:

            adj, x = adj.to(device), x.to(device)
            adj_c = 1 - adj
            assert adj_c.is_cuda
            # Apply the model to obtain scores (forward pass)
            logits = model(x, adj_c)
            # compute the probability of action for each instance
            actions = logits.max(dim=-1)[1]
            action_masks = F.one_hot(actions, args.n_hosts).type(torch.float32)
            assert action_masks.is_cuda
            assert action_masks.size() == (args.batch, args.n_phones, args.n_hosts)
            log_probs = (action_masks * F.log_softmax(logits, dim=-1)).sum(dim=-1).sum(dim=-1, keepdim=True)
            assert log_probs.is_cuda
            assert log_probs.requires_grad is True
            assert log_probs.size() == (args.batch, 1)
            # Compute the return
            risk_after = torch.matmul(torch.matmul(action_masks.transpose(dim0=1, dim1=2),
                                                   adj_c), action_masks)
            risk_after = risk_after.diagonal(dim1=1, dim2=2).sum(dim=-1, keepdim=True)
            risk_before = torch.matmul(torch.matmul(x.transpose(dim0=1, dim1=2),
                                                    adj_c), x)
            risk_before = risk_before.diagonal(dim1=1, dim2=2).sum(dim=-1, keepdim=True)
            ret = risk_before - risk_after
            assert ret.size() == (args.batch, 1)
            if args.baseline:
                if steps_done == 0:
                    M = ret.mean()
                else:
                    M = args.beta * M + (1 - args.beta) * ret.mean()
                b = M  # baseline
            else:
                b = 0
            # back-propagation
            reinforce = (ret - b) * log_probs / args.scale
            loss = -reinforce.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            steps_done += 1

        # Monitor results every report interval
        # if iter_idx % args.rep_intv == 0:
        log_cont.update({'ret_{}'.format(epoch): ret.mean().cpu().item()})
        ret_list.append(ret.mean().cpu().item())
        scalar = ret.mean().cpu().item()
        tr_writer.add_scalar("data/return", scalar, global_step=epoch)
        # # Save
        # torch.save({
        #     "epoch": epoch,
        #     "best_val_ret": best_val_ret,
        #     "model": model.state_dict(),
        #     "optimizer": optimizer.state_dict(),
        # }, checkpoint_file)

        # Validate results every validation interval
        # if iter_idx % args.val_intv == 0:
        model.eval()
        # Apply forward pass to compute the returns for each of the val batches
        with torch.no_grad():
            logits = model(val_x, val_adj_c)
            actions = logits.max(dim=-1)[1]
            action_masks = F.one_hot(actions, args.n_hosts).type(torch.float32)
            assert action_masks.is_cuda
        # Compute the return
        risk = torch.matmul(torch.matmul(action_masks.transpose(dim0=1, dim1=2),
                                         val_adj_c), action_masks)
        risk = risk.diagonal(dim1=1, dim2=2).sum(dim=-1, keepdim=True)
        val_ret = risk_init - risk
        log_cont.update({'val_ret_{}'.format(epoch): val_ret.mean().cpu().item()})
        val_ret_list.append(val_ret.mean().cpu().item())
        val_scalar = val_ret.mean().cpu().item()
        val_writer.add_scalar("data/return", val_scalar, global_step=epoch)
        model.train()

        if val_ret.mean() > best_val_ret:
            best_val_ret = val_ret.mean()
            # Save
            torch.save({
                "model": model.state_dict(),
            }, checkpoint_file)

    log_handle = open(log_path, 'wb')
    pickle.dump(log_cont, log_handle)
    log_handle.close()

    mpl.use('Agg')
    fig = plt.figure()
    plt.plot(ret_list)
    plt.plot(ret_list, "g--", label="train")
    plt.plot(val_ret_list, "-r", label="validation")
    plt.xlabel('training epoch')
    plt.ylabel('return')
    plt.legend(loc='best')
    fig.savefig(os.path.join(args.log_dir, 'plot_reinforce_tr_{}.png'.format(args.name_idx)))


def test(args):
    # CUDA_LAUNCH_BLOCKING = 1
    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ModelClass(args).to(device)

    model_name_list = ['3_1']

    n_phones_list = [20, 30, 40, 50, 60]
    risk_rl_list = []
    risk_rnd_list = []
    risk_qp_list = []

    log_path = os.path.join(args.log_dir, "log_reinforce_test-3_1.pickle")
    log_handle = open(log_path, 'wb')
    pickle.dump({}, log_handle)
    log_handle = open(log_path, 'rb')
    log_cont = pickle.load(log_handle)

    for n_phones in n_phones_list:
        # Generate adj matrix
        test_adj = torch.triu(torch.rand(args.test_batch, n_phones, n_phones), diagonal=1)
        test_adj = test_adj + test_adj.transpose(dim0=1, dim1=2)
        test_adj = test_adj + torch.eye(n_phones)
        test_adj_c = (1 - test_adj).to(device)
        test_adj_c_np = (1 - test_adj).numpy()

        # Generate random allocation matrix
        test_hot_idx = torch.randint(high=args.n_hosts, size=(args.test_batch * n_phones,))  # host index
        test_x = F.one_hot(test_hot_idx, num_classes=args.n_hosts).reshape(
            args.test_batch, n_phones, args.n_hosts).type(torch.float32).to(device)

        test_x_np = torch.rand(args.test_batch * n_phones * args.n_hosts, ).numpy()

        # random
        risk_rnd = torch.matmul(torch.matmul(test_x.transpose(dim0=1, dim1=2),
                                             test_adj_c), test_x)
        assert risk_rnd.diagonal(dim1=1, dim2=2).size() == (args.test_batch, args.n_hosts)
        assert risk_rnd.requires_grad is False
        risk_rnd = risk_rnd.diagonal(dim1=1, dim2=2).sum(dim=-1).mean()

        risk_rnd_list.append(risk_rnd.cpu().item())
        print('risk_rnd_{}: '.format(n_phones), risk_rnd.cpu().item())
        log_cont.update({'risk_rnd_{}: '.format(n_phones): risk_rnd.cpu().item()})

        # RL algorithm
        best_risk = float('inf')
        for _m, model_name in enumerate(model_name_list):

            checkpoint_file = os.path.join(args.save_dir,
                                           "checkpoint_{}.path".format(model_name_list[_m]))
            res = torch.load(checkpoint_file)
            model.load_state_dict(res["model"])
            model.eval()

            with torch.no_grad():
                logits = model(test_x, test_adj_c)
                actions = logits.max(dim=-1)[1]
                action_masks = F.one_hot(actions, args.n_hosts).type(torch.float32)
                assert action_masks.is_cuda
            # Compute the return
            risk_rl = torch.matmul(torch.matmul(action_masks.transpose(dim0=1, dim1=2),
                                                test_adj_c), action_masks)
            assert risk_rl.diagonal(dim1=1, dim2=2).size() == (args.test_batch, args.n_hosts)
            assert risk_rl.requires_grad is False
            risk_rl = risk_rl.diagonal(dim1=1, dim2=2).sum(dim=-1).mean()
            if risk_rl < best_risk:
                best_risk = risk_rl

            print('risk_rl_{}_{}: '.format(n_phones, model_name), risk_rl.cpu().item())
            log_cont.update({'risk_rl_{}_{}'.format(n_phones, model_name): risk_rl.cpu().item()})

        print('best_risk_rl_{}: '.format(n_phones), best_risk.cpu().item())
        log_cont.update({'best_risk_rl_{}'.format(n_phones): best_risk.cpu().item()})
        risk_rl_list.append(best_risk.cpu().item())

        # QP
        arr = diags([1] * n_phones).toarray()
        M1 = np.concatenate([arr] * 5, axis=1)
        # print(M1.shape)
        v1 = np.ones((n_phones, 1))
        # print(v1.shape)

        bnds = tuple([(0, None)] * n_phones * 5)

        cons1 = ({'type': 'eq', 'fun': cons_fun1, 'args': [v1, M1]},
                 {'type': 'ineq', 'fun': cons_fun3})

        cons2 = ({'type': 'eq', 'fun': cons_fun1, 'args': [v1, M1]})

        risk_list = []
        for _i in range(args.test_batch):
            adj_c = test_adj_c_np[_i]
            idx1 = _i * n_phones * args.n_hosts
            idx2 = (_i + 1) * n_phones * args.n_hosts
            x0 = test_x_np[idx1:idx2]

            arr = [adj_c for _ in range(5)]
            A_c = block_diag(arr).toarray()

            # Quadratic programming
            best_risk_qp = np.inf

            res1 = minimize(loss, x0, jac=jac, hess=hess, args=A_c, method='trust-constr', constraints=cons1)
            if res1.success:
                x = res1.x.reshape(5, n_phones)
                s = np.zeros((n_phones, 5))
                s[np.arange(len(x.T)), x.T.argmax(1)] = 1
                risk = np.dot(s.T.reshape(1, -1), np.dot(A_c, s.T.reshape(-1, 1))).item()
                if min(res1.fun, risk) < best_risk_qp:
                    best_risk_qp = min(res1.fun, risk)

            if n_phones <= 60:
                res3 = minimize(loss, x0, jac=jac, args=A_c, method='SLSQP', bounds=bnds, constraints=cons2)
                if res3.success:
                    x = res3.x.reshape(5, n_phones)
                    s = np.zeros((n_phones, 5))
                    s[np.arange(len(x.T)), x.T.argmax(1)] = 1
                    risk = np.dot(s.T.reshape(1, -1), np.dot(A_c, s.T.reshape(-1, 1))).item()
                    if min(res3.fun, risk) < best_risk_qp:
                        best_risk_qp = min(res3.fun, risk)

                res4 = minimize(loss, x0, jac=jac, args=A_c, method='SLSQP', constraints=cons1)
                if res4.success:
                    x = res4.x.reshape(5, n_phones)
                    s = np.zeros((n_phones, 5))
                    s[np.arange(len(x.T)), x.T.argmax(1)] = 1
                    risk = np.dot(s.T.reshape(1, -1), np.dot(A_c, s.T.reshape(-1, 1))).item()
                    if min(res4.fun, risk) < best_risk_qp:
                        best_risk_qp = min(res4.fun, risk)

            if best_risk_qp != np.inf:
                risk_list.append(best_risk_qp)

        best_risk_qp_avg = np.mean(risk_list)
        risk_qp_list.append(best_risk_qp_avg)
        print('best_risk_qp_avg_{}: '.format(n_phones), best_risk_qp_avg)
        log_cont.update({'best_risk_qp_avg_{}: '.format(n_phones): best_risk_qp_avg})

    log_handle = open(log_path, 'wb')
    pickle.dump(log_cont, log_handle)
    log_handle.close()

    mpl.use('Agg')
    fig = plt.figure()
    plt.plot(n_phones_list, risk_rl_list, "g--", label="RL")
    plt.plot(n_phones_list, risk_rnd_list, "-r", label="random")
    plt.plot(n_phones_list, risk_qp_list, "-b", label="QP")
    # plt.plot(n_phones_list, risk_hdf_list, "-b", label="HDF")
    # plt.plot(n_phones_list, risk_mcf_list, "-g", label="MCF")
    # # plt.plot(T, power_smooth, "g--", label="QP")
    plt.ylabel('potential risk')
    plt.xlabel('n_phones')
    plt.legend(loc='best')
    fig.savefig(os.path.join(args.log_dir, 'plot_reinforce_test-3_1.png'))


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


class graphs(Dataset):
    def __init__(self, hdf5_file_path):
        self.hdf5_file = h5py.File(hdf5_file_path, mode='r')
        self.transform = transforms.ToTensor()
        self.adj_shp = self.hdf5_file["adj"][0].shape
        self.x_shp = self.hdf5_file["x"][0].shape

    def __len__(self):
        return len(self.hdf5_file["adj"])

    def __getitem__(self, idx):
        adj = self.hdf5_file["adj"][idx]
        x = self.hdf5_file["x"][idx]
        adj = self.transform(adj).squeeze()
        x = self.transform(x).squeeze()
        return adj, x


class Normalization(nn.Module):

    def __init__(self, embed_dim):
        super(Normalization, self).__init__()

        self.normalizer = nn.BatchNorm1d(embed_dim, affine=True)

        # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
        # self.init_parameters()

    # def init_parameters(self):
    #
    #     for name, param in self.named_parameters():
    #         stdv = 1. / math.sqrt(param.size(-1))
    #         param.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())


class ModelClass(nn.Module):
    def __init__(self, args):
        super(ModelClass, self).__init__()

        self.n_layers = args.n_layers

        self.theta1 = nn.Linear(args.n_hosts, args.embed_dim, bias=False)
        self.pre_pooling = nn.Linear(args.embed_dim, args.embed_dim, bias=True)
        self.theta2 = nn.Linear(args.embed_dim, args.embed_dim, bias=False)
        self.norm = Normalization(args.embed_dim)

        self.last = nn.Linear(args.embed_dim, args.n_hosts, bias=False)

    def forward(self, x, adj_c):
        """
        :param x: (batch, graph_size, node_dim) input node features
        """
        # for _l in range(self.n_layers):
        #     h_part1 = self.theta1(x)
        #
        #     if _l != 0:
        #         h = F.relu(self.pre_pooling(h))
        #         h_part2 = self.theta2(torch.matmul(adj_c, h))
        #
        #     if _l != 0:
        #         h = F.relu(h_part1 + h_part2)
        #     else:
        #         h = F.relu(h_part1)
        #
        #   return h

        h = self.theta1(x)

        for _l in range(self.n_layers):
            mu = F.relu(self.pre_pooling(h))
            mu = self.theta2(torch.matmul(adj_c, mu))
            h = F.relu(mu+h)

        logits = self.last(h)

        return logits


# def model_loss(args, model):
#     loss = 0
#     for name, param in model.named_parameters():
#         if "weight" in name:
#             loss += torch.sum(param**2)
#     return loss * args.l2_reg


if __name__ == "__main__":
    main()
