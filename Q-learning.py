
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

parser.add_argument("--mode", type=str,
                    default="train",
                    choices=["train", "test"],
                    help="Run mode")

# parser.add_argument("--data_dir", type=str,
#                     default="/home/pasha/scratch/datasets/graphs",
#                     help="")

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
                    default=2000,
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
                    default=62,
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

parser.add_argument("--name_idx", type=str,
                    default='0',
                    help="")


def main():
    """The main function."""
    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test(args)
    else:
        raise ValueError("Unknown run mode \"{}\"".format(args.mode))


def train(args):
    # random.seed(7)
    # torch.manual_seed(7)

    # CUDA_LAUNCH_BLOCKING = 1
    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_net = DQN(args).to(device)
    target_net = DQN(args).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    data_loss = data_criterion(args)

    if args.optim == 'adam':
        optimizer = optim.Adam(policy_net.parameters(), lr=args.lr)
    elif args.optim == 'rmsprop':
        optimizer = optim.RMSprop(policy_net.parameters(), lr=args.lr)

    Transition = namedtuple('Transition',
                            ('state', 'adj_c', 'phone', 'action', 'next_state', 'acc_reward', 'D'))

    capacity = args.n_episodes * args.n_phones
    memory = ReplayMemory(capacity)

    q = torch.tensor([1/10, 1/10, 2/10, 3/10, 3/10]).type(torch.float32).to(device)

    eps = 1e-5

    ################# random validation dataset #####################

    # Path to valid data
    # val_file_path = os.path.join(args.data_dir, 'val.h5')

    # val_adj = torch.rand(args.val_batch, args.n_phones, args.n_phones)
    # val_adj = (val_adj + val_adj.transpose(dim0=1, dim1=2)) / 2
    # for _j in range(args.n_phones):
    #     val_adj[:, _j, _j] = 1
    # val_adj_c = (1 - val_adj)
    val_adj = torch.triu(torch.rand(args.val_batch, args.n_phones, args.n_phones), diagonal=1)
    val_adj = val_adj + val_adj.transpose(dim0=1, dim1=2)
    val_adj = val_adj + torch.eye(args.n_phones)
    val_adj_c = (1 - val_adj)
    # Generate random allocation matrix for val dataset
    val_hot_idx = torch.randint(high=args.n_hosts, size=(args.val_batch * args.n_phones, ))  # host index
    val_x = F.one_hot(val_hot_idx, num_classes=args.n_hosts).reshape(
        args.val_batch, args.n_phones, args.n_hosts).type(torch.float32)

    risk_init = torch.matmul(torch.matmul(val_x.transpose(dim0=1, dim1=2).to(device),
                                          val_adj_c.to(device)), val_x.to(device))
    risk_init = risk_init.diagonal(dim1=1, dim2=2).sum(dim=-1)

    print('risk_init: ', risk_init.mean())

    p_init = (val_x.sum(dim=1) / args.n_phones).to(device) + eps
    pq_init = (p_init * torch.log2(p_init / q.reshape(1, args.n_hosts))).sum(dim=-1)

    cost_init = risk_init + args.reg * pq_init

    print('pq_init: ', pq_init.mean())

    print('cost_init: ', cost_init.mean())

    #################################################################

    # Create log directory and save directory if it does not exist
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Create summary writer
    val_writer = SummaryWriter(
        log_dir=os.path.join(args.log_dir, "valid_{}".format(args.name_idx)))
    train_writer = SummaryWriter(
        log_dir=os.path.join(args.log_dir, "train_{}".format(args.name_idx)))
    # Prepare checkpoint file
    checkpoint_file = os.path.join(args.save_dir, "checkpoint_{}.path".format(args.name_idx))

    n_phones = args.n_phones
    n_hosts = args.n_hosts
    eps_end = args.eps_end
    eps_start = args.eps_start
    eps_decay = args.eps_decay
    delay = args.delay
    replay_batch = args.replay_batch
    gamma = args.gamma
    val_batch = args.val_batch
    scale = args.scale

    # Initialize training
    steps_done = 0
    best_ret = 0
    ret_list = []
    # cons = ({'type': 'eq', 'fun': cons_fun1},
    #         {'type': 'ineq', 'fun': cons_fun2})

    # Training loop
    for episode in tqdm(range(args.n_episodes)):
        stack_xa = StackXAction(maxsize=delay)
        stack_reward = StackReward(maxsize=delay)

        # Generate random graph
        # adj = torch.rand(1, args.n_phones, args.n_phones)
        # adj = (adj + adj.transpose(dim0=1, dim1=2)) / 2
        # for _j in range(args.n_phones):
        #     adj[:, _j, _j] = 1
        # adj_c = (1 - adj).to(device)
        adj = torch.triu(torch.rand(1, args.n_phones, args.n_phones), diagonal=1)
        adj = adj + adj.transpose(dim0=1, dim1=2)
        adj = adj + torch.eye(args.n_phones)
        adj_c = (1 - adj).to(device)
        # Generate random allocation matrix
        hot_idx = torch.randint(high=args.n_hosts, size=(1 * args.n_phones, ))  # host index
        x = F.one_hot(hot_idx, num_classes=args.n_hosts).reshape(
            1, args.n_phones, args.n_hosts).type(torch.float32).to(device)

        # while True:
        #     # Generate random graph
        #     adj = torch.triu(torch.rand(1, args.n_phones, args.n_phones), diagonal=1)
        #     adj = adj + adj.transpose(dim0=1, dim1=2)
        #     adj = adj + torch.eye(args.n_phones)
        #     adj_c = (1 - adj).to(device)
        #     adj_c_np = (1 - adj).squeeze(dim=0).numpy()
        #     # Generate random allocation matrix
        #     x_np = torch.rand(args.n_phones * args.n_hosts, 1).numpy()
        #
        #     res = optimize.minimize(loss_fun, x_np, args=adj_c_np, method='SLSQP', constraints=cons)
        #
        #     if res.message == 'Optimization terminated successfully.':
        #         break
        #
        # s_idx = np.argmax(res.x.reshape(args.n_phones, args.n_hosts), axis=-1)
        # s = np.zeros((args.n_phones, args.n_hosts))
        # s[np.arange(n_phones), s_idx] = 1
        #
        # x = torch.from_numpy(s).unsqueeze(dim=0).type(torch.float32).to(device)

        risk_bf = torch.matmul(torch.matmul(x.transpose(dim0=1, dim1=2), adj_c), x)
        risk_bf = risk_bf.diagonal(dim1=1, dim2=2).sum(dim=-1)

        p_bf = x.sum(dim=1) / args.n_phones + eps
        pq_bf = (p_bf * torch.log2(p_bf / q.reshape(1, args.n_hosts))).sum(dim=-1)

        for _i in range(n_phones):
            # select action
            sample = random.random()
            eps_threshold = eps_end + (eps_start - eps_end) * \
                            math.exp(-1. * steps_done / eps_decay)
            # steps_done += 1
            if sample > eps_threshold:
                with torch.no_grad():
                    # t.max(1) will return largest column value of each row.
                    # second column on max result is index of where max element was
                    # found, so we pick action with the larger expected reward.
                    phone = torch.tensor([_i], device=device)
                    action = policy_net(x, adj_c, phone).max(1)[1].view(1, 1)
            else:
                action = \
                    torch.tensor([[random.randrange(n_hosts)]], device=device, dtype=torch.long)
            assert action.is_cuda
            stack_xa.put((x, action))

            # Compute the reward
            x_i = F.one_hot(action, num_classes=n_hosts).type(torch.float32).to(device)
            x[:, _i, :] = x_i.squeeze(dim=1)
            risk_af = torch.matmul(torch.matmul(x.transpose(dim0=1, dim1=2), adj_c), x)
            risk_af = risk_af.diagonal(dim1=1, dim2=2).sum(dim=-1)
            p_af = x.sum(dim=1) / args.n_phones + eps
            pq_af = (p_af * torch.log2(p_af / q.reshape(1, args.n_hosts))).sum(dim=-1)
            cost_bf = risk_bf + args.reg * pq_bf
            cost_af = risk_af + args.reg * pq_af
            reward = (cost_bf - cost_af).item()
            assert risk_af.is_cuda
            assert risk_af.requires_grad is False
            stack_reward.put(reward)

            if _i+1 < n_phones:
                next_state = x
                risk_bf = risk_af
                pq_bf = pq_af
            else:
                next_state = None

            if _i+1 >= delay:
                state, action = stack_xa.get(0)
                assert state.is_cuda
                acc_reward = 0
                for _j in range(delay):
                    acc_reward += stack_reward.get(_j)
                acc_reward = torch.tensor([acc_reward], device=device)
                phone = torch.tensor([_i+1-delay], device=device)
                D = torch.tensor([delay], device=device)
                memory.push(state, adj_c, phone, action, next_state, acc_reward, D)
                steps_done += 1

                # Optimize model
                if len(memory) >= replay_batch:
                    transitions = memory.sample(replay_batch)
                    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
                    # detailed explanation). This converts batch-array of Transitions
                    # to Transition of batch-arrays.
                    batch = Transition(*zip(*transitions))

                    # Compute a mask of non-rl states and concatenate the batch elements
                    # (a rl state would've been the one after which simulation ended)
                    non_rl_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                            batch.next_state)), device=device, dtype=torch.bool)
                    non_rl_next_states = torch.cat([s for s in batch.next_state
                                                       if s is not None])
                    non_rl_next_adj_c = torch.cat([s for _j, s in enumerate(batch.adj_c)
                                                      if non_rl_mask[_j].item() is True])
                    non_rl_D = torch.cat([s for _j, s in enumerate(batch.D)
                                             if non_rl_mask[_j].item() is True])
                    non_rl_next_phone = torch.cat([s for _j, s in enumerate(batch.phone)
                                                      if non_rl_mask[_j].item() is True])
                    assert non_rl_next_phone.size() == non_rl_D.size()
                    non_rl_next_phone = non_rl_next_phone + non_rl_D
                    assert len(non_rl_next_states) == len(non_rl_next_adj_c)
                    assert non_rl_next_adj_c.is_cuda
                    state_batch = torch.cat(batch.state)
                    adj_c_batch = torch.cat(batch.adj_c)
                    phone_batch = torch.cat(batch.phone)  # (replay_batch, )
                    action_batch = torch.cat(batch.action)
                    acc_reward_batch = torch.cat(batch.acc_reward)  # (replay_batch, )
                    assert adj_c_batch.is_cuda

                    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
                    # columns of actions taken. These are the actions which would've been taken
                    # for each batch state according to policy_net
                    state_action_values = policy_net(state_batch, adj_c_batch, phone_batch).gather(1, action_batch)
                    # Compute V(s_{t+1}) for all next states.
                    # Expected values of actions for non_rl_next_states are computed based
                    # on the "older" target_net; selecting their best reward with max(1)[0].
                    # This is merged based on the mask, such that we'll have either the expected
                    # state value or 0 in case the state was rl.
                    next_state_values = torch.zeros(replay_batch, device=device)
                    next_state_values[non_rl_mask] = target_net(non_rl_next_states,
                                                                   non_rl_next_adj_c ,
                                                                   non_rl_next_phone).max(1)[0].detach()

                    # Compute the expected Q values
                    assert next_state_values.shape == acc_reward_batch.shape
                    expected_state_action_values = next_state_values * gamma + acc_reward_batch / scale

                    # Compute Huber loss
                    # loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
                    loss = data_loss(state_action_values,
                                     expected_state_action_values.unsqueeze(1))

                    assert state_action_values.size() == expected_state_action_values.unsqueeze(1).size()

                    if len(memory) == replay_batch:
                        # Monitor train loss every episode
                        train_writer.add_scalar("data/q1", state_action_values.mean(), global_step=episode)
                        train_writer.add_scalar("data/q2", next_state_values.mean(), global_step=episode)
                        train_writer.add_scalar("data/reward", acc_reward_batch.mean(), global_step=episode)

                        ##################### validation result #########################

                        policy_net.eval()

                        v_adj_c = val_adj_c.clone().to(device)
                        v_x = val_x.clone().to(device)

                        for _j in range(n_phones):
                            with torch.no_grad():
                                phone_set = torch.ones(val_batch).type(torch.int).to(device) * _i

                                # select action
                                action = policy_net(v_x, v_adj_c, phone_set).max(1)[1]

                                # Compute the reward
                                v_x_j = F.one_hot(action, num_classes=args.n_hosts).type(torch.float32).to(device)
                                v_x[:, _j, :] = v_x_j
                                assert v_x.is_cuda

                        risk_rl = torch.matmul(torch.matmul(v_x.transpose(dim0=1, dim1=2), v_adj_c), v_x)
                        assert risk_rl.diagonal(dim1=1, dim2=2).size() == (args.val_batch, args.n_hosts)
                        assert risk_rl.requires_grad is False
                        risk_rl = risk_rl.diagonal(dim1=1, dim2=2).sum(dim=-1)

                        p_rl = v_x.sum(dim=1) / args.n_phones + eps
                        pq_rl = (p_rl * torch.log2(p_rl / q.reshape(1, args.n_hosts))).sum(dim=-1)

                        cost_rl = risk_rl + args.reg * pq_rl

                        ret = (cost_init - cost_rl).mean().item()  # return per episode
                        ret_list.append(ret)

                        # Monitor validation results every episode
                        val_writer.add_scalar("data/return", ret, global_step=episode)

                        policy_net.train()

                        #################################################################

                    # Optimize the model
                    optimizer.zero_grad()
                    loss.backward()
                    if args.clamp:
                        for param in policy_net.parameters():
                            param.grad.data.clamp_(-1, 1)
                    optimizer.step()

                    # Update the target network, copying all weights and biases in DQN
                    if steps_done % args.target_update == 0:
                        target_net.load_state_dict(policy_net.state_dict())

        if len(memory) > replay_batch:
            # Monitor train loss every episode
            train_writer.add_scalar("data/q1", state_action_values.mean(), global_step=episode)
            train_writer.add_scalar("data/q2", next_state_values.mean(), global_step=episode)
            train_writer.add_scalar("data/reward", acc_reward_batch.mean(), global_step=episode)

            ##################### validation result #########################

            policy_net.eval()

            v_adj_c = val_adj_c.clone().to(device)
            v_x = val_x.clone().to(device)

            for _j in range(n_phones):
                with torch.no_grad():
                    phone_set = torch.ones(val_batch).type(torch.int).to(device) * _i

                    # select action
                    action = policy_net(v_x, v_adj_c, phone_set).max(1)[1]

                    # Compute the reward
                    v_x_j = F.one_hot(action, num_classes=args.n_hosts).type(torch.float32).to(device)
                    v_x[:, _j, :] = v_x_j
                    assert v_x.is_cuda

            risk_rl = torch.matmul(torch.matmul(v_x.transpose(dim0=1, dim1=2), v_adj_c), v_x)
            assert risk_rl.diagonal(dim1=1, dim2=2).size() == (args.val_batch, args.n_hosts)
            assert risk_rl.requires_grad is False
            risk_rl = risk_rl.diagonal(dim1=1, dim2=2).sum(dim=-1)

            p_rl = v_x.sum(dim=1) / args.n_phones + eps
            pq_rl = (p_rl * torch.log2(p_rl / q.reshape(1, args.n_hosts))).sum(dim=-1)

            cost_rl = risk_rl + args.reg * pq_rl

            ret = (cost_init - cost_rl).mean().item()  # return per episode
            ret_list.append(ret)

            # Monitor validation results every episode
            val_writer.add_scalar("data/return", ret, global_step=episode)

            policy_net.train()

            if ret >= best_ret:
                best_ret = ret
                torch.save({"model": policy_net.state_dict()}, checkpoint_file)
                mpl.use('Agg')
                fig = plt.figure()
                center = [1, 2, 3, 4, 5]
                plt.bar(center, v_x.sum(dim=1).mean(dim=0).cpu().numpy(),
                        align='center', width=0.5)
                plt.xlabel('hosts')
                plt.ylabel('num_phones_avg')
                fig.savefig(os.path.join(args.log_dir, 'plot_val_{}.png'.format(args.name_idx)))

        if episode >= args.delay_trigger and episode % 10 == 0:
            delay += 2

        #################################################################

    print('delay:', delay)

    # mpl.use('Agg')
    # fig = plt.figure()
    # plt.plot(ret_list)
    # plt.ylabel('data/return')
    # fig.savefig(os.path.join(args.log_dir, 'plot_{}.png'.format(args.name_idx)))


def test(args):
    # CUDA_LAUNCH_BLOCKING = 1
    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_net1 = DQN(args).to(device)

    if args.constrained:
        model_name_list = ['32_30', '32_51']
        # model_name_list = ['32_30']

        n_phones_list = [20, 30, 40, 50, 60]
        risk_rl_list = []
        risk_qp_list = []

        log_path = os.path.join(args.log_dir, "log_rl_qp_11233-32_30-32_51.pickle")
        log_handle = open(log_path, 'wb')
        pickle.dump({}, log_handle)
        log_handle = open(log_path, 'rb')
        log_cont = pickle.load(log_handle)

        for n_phones in n_phones_list:
            # Generate adj matrix
            test_adj = torch.triu(torch.rand(args.test_batch, n_phones, n_phones), diagonal=1)
            test_adj = test_adj + test_adj.transpose(dim0=1, dim1=2)
            test_adj = test_adj + torch.eye(n_phones)
            test_adj_c = (1 - test_adj)
            test_adj_c_np = (1 - test_adj).numpy()

            # Generate random allocation matrix
            test_hot_idx = torch.randint(high=args.n_hosts, size=(args.test_batch * n_phones,))  # host index
            test_x = F.one_hot(test_hot_idx, num_classes=args.n_hosts).reshape(
                args.test_batch, n_phones, args.n_hosts).type(torch.float32)

            test_x_np = torch.rand(args.test_batch * n_phones * args.n_hosts, ).numpy()

            # RL algorithm
            best_risk = float('inf')
            for _m, model_name in enumerate(model_name_list):
                adj_c = torch.zeros(args.test_batch, n_phones, n_phones)
                _ = adj_c.copy_(test_adj_c)
                adj_c = adj_c.to(device)
                x = torch.zeros(args.test_batch, n_phones, args.n_hosts)
                _ = x.copy_(test_x)
                x = x.to(device)

                checkpoint_file = os.path.join(args.save_dir,
                                               "checkpoint_{}.path".format(model_name_list[_m]))
                res = torch.load(checkpoint_file)
                policy_net1.load_state_dict(res["model"])
                policy_net1.eval()

                for _i in range(n_phones):
                    with torch.no_grad():
                        phone_set = torch.ones(args.test_batch).type(torch.int).to(device) * _i
                        # select action
                        action1 = policy_net1(x, adj_c, phone_set).max(1)[1]
                        # Compute the reward
                        x_i = F.one_hot(action1, num_classes=args.n_hosts).type(torch.float32).to(device)
                        x[:, _i, :] = x_i
                        assert x.is_cuda
                risk_rl = torch.matmul(torch.matmul(x.transpose(dim0=1, dim1=2), adj_c), x)
                assert risk_rl.diagonal(dim1=1, dim2=2).size() == (args.test_batch, args.n_hosts)
                assert risk_rl.requires_grad is False
                risk_rl = risk_rl.diagonal(dim1=1, dim2=2).sum(dim=-1).mean()
                if risk_rl < best_risk:
                    best_risk = risk_rl
                    mpl.use('Agg')
                    fig = plt.figure()
                    center = [1, 2, 3, 4, 5]
                    plt.bar(center, x.sum(dim=1).mean(dim=0).cpu().numpy(),
                            align='center', width=0.5)
                    plt.xlabel('hosts')
                    plt.ylabel('num_phones_avg')
                    fig.savefig(os.path.join(args.log_dir,
                                             'bar_rl_qp_11233_{}.png'.format(n_phones)))

                print('risk_rl_{}_{}: '.format(n_phones, model_name), risk_rl.cpu().item())
                log_cont.update({'risk_rl_{}_{}'.format(n_phones, model_name): risk_rl.cpu().item()})

            print('best_risk_rl_{}: '.format(n_phones), best_risk.cpu().item())
            log_cont.update({'best_risk_rl_{}'.format(n_phones): best_risk.cpu().item()})
            risk_rl_list.append(best_risk.cpu().item())

            # QP algorithm
            q = np.array([1 / 10, 1 / 10, 2 / 10, 3 / 10, 3 / 10]).reshape(5, ) * n_phones

            arr = diags([1] * n_phones).toarray()
            M1 = np.concatenate([arr] * 5, axis=1)
            # print(M1.shape)
            v1 = np.ones((n_phones, 1))
            # print(v1.shape)

            e = np.ones((1, n_phones))
            arr = [e for _ in range(5)]
            M2 = block_diag(arr).toarray()
            # print(M2.shape)
            v2 = q.reshape(5, 1)
            # print(v2.shape)

            I = -np.eye(n_phones * 5)
            G = np.vstack((M2, I))
            # print(G.shape)
            h = np.vstack((q.reshape((5, 1)), np.zeros((n_phones * 5, 1))))
            # print(h.shape)

            bnds = tuple([(0, None)] * n_phones * 5)

            cons1 = ({'type': 'eq', 'fun': cons_fun1, 'args': [v1, M1]},
                     {'type': 'ineq', 'fun': cons_fun2, 'args': [v2, M2]},
                     {'type': 'ineq', 'fun': cons_fun3})

            cons2 = ({'type': 'eq', 'fun': cons_fun1, 'args': [v1, M1]},
                     {'type': 'ineq', 'fun': cons_fun2, 'args': [v2, M2]})

            cons3 = ({'type': 'eq', 'fun': cons_fun1, 'args': [v1, M1]},
                     {'type': 'ineq', 'fun': cons_fun4, 'args': [h, G]})

            risk_list = []
            for _i in range(args.test_batch):
                adj_c = test_adj_c_np[_i]
                idx1 = _i * n_phones * args.n_hosts
                idx2 = (_i+1) * n_phones * args.n_hosts
                x0 = test_x_np[idx1:idx2]

                arr = [adj_c for _ in range(5)]
                A_c = block_diag(arr).toarray()

                # Quadratic programming
                best_risk_qp = np.inf

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
                    if min(res1.fun, risk) < best_risk_qp:
                        best_risk_qp = min(res1.fun, risk)

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
                    if min(res2.fun, risk) < best_risk_qp:
                        best_risk_qp = min(res2.fun, risk)

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
                        if min(res3.fun, risk) < best_risk_qp:
                            best_risk_qp = min(res3.fun, risk)

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
                        if min(res4.fun, risk) < best_risk_qp:
                            best_risk_qp = min(res4.fun, risk)

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
                        if min(res5.fun, risk) < best_risk_qp:
                            best_risk_qp = min(res5.fun, risk)

                risk_list.append(best_risk_qp)

            best_risk_qp_avg = np.mean(risk_list)
            risk_qp_list.append(best_risk_qp_avg)
            print('best_risk_qp_avg_{}: '.format(n_phones), best_risk_qp_avg)
            log_cont.update({'best_risk_qp_avg_{}: '.format(n_phones): best_risk_qp_avg})

        log_handle = open(log_path, 'wb')
        pickle.dump(log_cont, log_handle)
        log_handle.close()

        # T = np.array(n_phones_list)
        # power = np.array(risk_qp_list)
        # xnew = np.linspace(T.min(), T.max(), 200)
        # power_smooth = spline(T, power, xnew)

        mpl.use('Agg')
        fig = plt.figure()
        plt.plot(n_phones_list, risk_rl_list, "g--", label="RL")
        plt.plot(n_phones_list, risk_qp_list, "-r", label="QP")
        # plt.plot(n_phones_list, risk_hdf_list, "-b", label="HDF")
        # plt.plot(n_phones_list, risk_mcf_list, "-g", label="MCF")
        # # plt.plot(T, power_smooth, "g--", label="QP")
        plt.ylabel('potential risk')
        plt.xlabel('n_phones')
        plt.legend(loc='best')
        fig.savefig(os.path.join(args.log_dir, 'plot_rl_qp_11233-32_30-32_51.png'))
    else:
        model_name_list = ['32_36']

        n_phones_list = [20, 30, 40, 50, 60]
        risk_rl_list = []
        risk_rnd_list = []

        log_path = os.path.join(args.log_dir, "log_rl_rnd-32_36.pickle")
        log_handle = open(log_path, 'wb')
        pickle.dump({}, log_handle)
        log_handle = open(log_path, 'rb')
        log_cont = pickle.load(log_handle)

        for n_phones in n_phones_list:
            # Generate adj matrix
            test_adj = torch.triu(torch.rand(args.test_batch, n_phones, n_phones), diagonal=1)
            test_adj = test_adj + test_adj.transpose(dim0=1, dim1=2)
            test_adj = test_adj + torch.eye(n_phones)
            test_adj_c = (1 - test_adj)

            # Generate random allocation matrix
            test_hot_idx = torch.randint(high=args.n_hosts, size=(args.test_batch * n_phones,))  # host index
            test_x = F.one_hot(test_hot_idx, num_classes=args.n_hosts).reshape(
                args.test_batch, n_phones, args.n_hosts).type(torch.float32)

            risk_rnd = torch.matmul(torch.matmul(test_x.transpose(dim0=1, dim1=2).to(device),
                                                 test_adj_c.to(device)), test_x.to(device))
            assert risk_rnd.diagonal(dim1=1, dim2=2).size() == (args.test_batch, args.n_hosts)
            assert risk_rnd.requires_grad is False
            risk_rnd = risk_rnd.diagonal(dim1=1, dim2=2).sum(dim=-1).mean()

            risk_rnd_list.append(risk_rnd.cpu().item())

            # RL algorithm
            best_risk = float('inf')
            for _m, model_name in enumerate(model_name_list):
                adj_c = torch.zeros(args.test_batch, n_phones, n_phones)
                _ = adj_c.copy_(test_adj_c)
                adj_c = adj_c.to(device)
                x = torch.zeros(args.test_batch, n_phones, args.n_hosts)
                _ = x.copy_(test_x)
                x = x.to(device)

                checkpoint_file = os.path.join(args.save_dir,
                                               "checkpoint_{}.path".format(model_name_list[_m]))
                res = torch.load(checkpoint_file)
                policy_net1.load_state_dict(res["model"])
                policy_net1.eval()

                for _i in range(n_phones):
                    with torch.no_grad():
                        phone_set = torch.ones(args.test_batch).type(torch.int).to(device) * _i
                        # select action
                        action1 = policy_net1(x, adj_c, phone_set).max(1)[1]
                        # Compute the reward
                        x_i = F.one_hot(action1, num_classes=args.n_hosts).type(torch.float32).to(device)
                        x[:, _i, :] = x_i
                        assert x.is_cuda
                risk_rl = torch.matmul(torch.matmul(x.transpose(dim0=1, dim1=2), adj_c), x)
                assert risk_rl.diagonal(dim1=1, dim2=2).size() == (args.test_batch, args.n_hosts)
                assert risk_rl.requires_grad is False
                risk_rl = risk_rl.diagonal(dim1=1, dim2=2).sum(dim=-1).mean()
                if risk_rl < best_risk:
                    best_risk = risk_rl
                    mpl.use('Agg')
                    fig = plt.figure()
                    center = [1, 2, 3, 4, 5]
                    plt.bar(center, x.sum(dim=1).mean(dim=0).cpu().numpy(),
                            align='center', width=0.5)
                    plt.xlabel('hosts')
                    plt.ylabel('num_phones_avg')
                    fig.savefig(os.path.join(args.log_dir,
                                             'bar_rl_qp_11233_{}.png'.format(n_phones)))

                print('risk_rl_{}_{}: '.format(n_phones, model_name), risk_rl.cpu().item())
                log_cont.update({'risk_rl_{}_{}'.format(n_phones, model_name): risk_rl.cpu().item()})

            print('best_risk_rl_{}: '.format(n_phones), best_risk.cpu().item())
            log_cont.update({'best_risk_rl_{}'.format(n_phones): best_risk.cpu().item()})
            risk_rl_list.append(best_risk.cpu().item())

            mpl.use('Agg')
            fig = plt.figure()
            plt.plot(n_phones_list, risk_rl_list, "g--", label="RL")
            plt.plot(n_phones_list, risk_rnd_list, "-r", label="QP")
            # plt.plot(n_phones_list, risk_hdf_list, "-b", label="HDF")
            # plt.plot(n_phones_list, risk_mcf_list, "-g", label="MCF")
            # # plt.plot(T, power_smooth, "g--", label="QP")
            plt.ylabel('potential risk')
            plt.xlabel('n_phones')
            plt.legend(loc='best')
            fig.savefig(os.path.join(args.log_dir, 'plot_rl_rnd-32_36.png'))


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


def data_criterion(args):
    """Returns the loss object based on the commandline argument for the data term
    """

    def square(x1, x2):
        return ((x1 - x2)**2).mean()

    if args.loss_type == "huber":
        data_loss = nn.SmoothL1Loss()
    elif args.loss_type == "square":
        data_loss = square

    return data_loss


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.Transition = namedtuple('Transition',
                                     ('state', 'adj_c', 'phone', 'action', 'next_state', 'acc_reward', 'D'))

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        else:
            raise ValueError("number of iterations exceeds {}".format(self.capacity))
        self.memory[self.position] = self.Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, replay_batch):
        return random.sample(self.memory, replay_batch)

    def __len__(self):
        return len(self.memory)


class StackReward(object):

    def __init__(self, maxsize):
        self.maxsize = maxsize
        self.memory = []

    def put(self, reward):
        """Saves a reward."""
        if len(self.memory) < self.maxsize:
            self.memory.append(reward)
        else:
            del self.memory[0]
            self.memory.append(reward)

    def get(self, index):
        return self.memory[index]

    def __len__(self):
        return len(self.memory)


class StackXAction(object):

    def __init__(self, maxsize):
        self.maxsize = maxsize
        self.memory = []

    def put(self, state):
        """Saves a reward."""
        if len(self.memory) < self.maxsize:
            self.memory.append(state)
        else:
            del self.memory[0]
            self.memory.append(state)

    def get(self, index):
        return self.memory[index]

    def __len__(self):
        return len(self.memory)


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


class GraphAttentionEncoder(nn.Module):
    def __init__(self, args):
        super(GraphAttentionEncoder, self).__init__()

        self.n_layers = args.n_layers

        self.theta1 = nn.Linear(args.n_hosts, args.embed_dim, bias=False)
        self.pre_pooling = nn.Linear(args.embed_dim, args.embed_dim, bias=True)
        self.theta2 = nn.Linear(args.embed_dim, args.embed_dim, bias=False)
        self.norm = Normalization(args.embed_dim)

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

        return h


class SingleHeadAggregation(nn.Module):
    def __init__(self, args):
        super(SingleHeadAggregation, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, h, adj_c, phone_set):
        """
        :param h: (batch, graph_size, embed_dim) embedding vector at hand
        :param phone_set: the set of phone indexes
        """
        g_bf_list = []
        g_af_list = []

        for _i, phone in enumerate(phone_set):
            phone = phone.item()
            batch, graph_size, embed_dim = h.size()

            ################## non-adjacent phones before ###############

            if phone > 0:
                adj_c_bf = adj_c[_i, phone, 0:phone].unsqueeze(dim=0)
                g_bf = torch.mm(adj_c_bf, h[_i, 0:phone, :])  # (1, embed_dim)
            else:
                g_bf = torch.zeros(1, embed_dim).to(self.device)

            assert g_bf.shape == (1, embed_dim)
            g_bf_list.append(g_bf)

            ################## non-adjacent phones after ###############

            if (phone+1) < graph_size:
                adj_c_af = adj_c[_i, phone, (phone+1):graph_size].unsqueeze(dim=0)
                g_af = torch.mm(adj_c_af, h[_i, (phone+1):graph_size, :])  # (1, embed_dim)
            else:
                g_af = torch.zeros(1, embed_dim).to(self.device)

            assert g_af.shape == (1, embed_dim)
            g_af_list.append(g_af)

        return torch.cat(g_bf_list, dim=0), torch.cat(g_af_list, dim=0)


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()

        self.shg = SingleHeadAggregation(args)

        self.left = nn.Linear(args.embed_dim, args.dec_ff_hidden, bias=False)
        self.right = nn.Linear(args.embed_dim, args.dec_ff_hidden, bias=False)
        self.action = nn.Linear(2*args.dec_ff_hidden, args.n_hosts, bias=False)

    def forward(self, h, adj_c, phone_set):
        """
        :param h: (1, graph_size, embed_dim) embedding vector at hand
        :param phone_set: the set of phone indexes
        """
        g_bf, g_af = self.shg(h, adj_c, phone_set)
        assert g_bf.is_cuda
        g_bf = self.left(g_bf)
        g_af = self.right(g_af)
        a = self.action(F.relu((torch.cat((g_bf, g_af), dim=-1))))  # action_values

        return a
    
    
class DQN(nn.Module):
    def __init__(self, args):
        super(DQN, self).__init__()
        
        self.encoder = GraphAttentionEncoder(args)
        self.decoder = Decoder(args)

    def forward(self, x, adj_c, phone_set):
        """
        :param x: (1, graph_size, node_dim) input node features
        :param phone_set: the set of phone indexes
        """
        h = self.encoder(x, adj_c)
        a = self.decoder(h, adj_c, phone_set)

        return a


if __name__ == "__main__":

    main()

