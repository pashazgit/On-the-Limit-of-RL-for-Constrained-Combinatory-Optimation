
import sys
# sys.path.insert(0, "./")
import os
import math
import random
import numpy as np
from collections import namedtuple
from itertools import count
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import h5py
import argparse
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

# parser.add_argument("--mode", type=str,
#                     default="train",
#                     choices=["train", "test"],
#                     help="Run mode")

# parser.add_argument("--data_dir", type=str,
#                     default="/home/pasha/scratch/datasets/graphs",
#                     help="")

parser.add_argument("--optim", type=str,
                    default="Adam",
                    help="")

parser.add_argument("--save_dir", type=str,
                    default="/home/pasha/scratch/jobs/output/RL-CO/saves",
                    help="Directory to save the best model")

parser.add_argument("--log_dir", type=str,
                    default="/home/pasha/scratch/jobs/output/RL-CO/logs",
                    help="Directory to save logs")

parser.add_argument("--rand_edge", type=float,
                    default=0.387,
                    help="")

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

parser.add_argument("--clip", type=str2bool,
                    default=True,
                    help="")

parser.add_argument("--clamp", type=str2bool,
                    default=True,
                    help="")

parser.add_argument("--target_update", type=int,
                    default=1,
                    help="")

# parser.add_argument("--init_stddev", type=float,
#                     default=1e-3,
#                     help="Learning rate (gradient step size)")

parser.add_argument("--replay_batch", type=int,
                    default=128,
                    help="")

parser.add_argument("--val_batch", type=int,
                    default=16,
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

parser.add_argument("--name_idx", type=int,
                    default=0,
                    help="")


def train():
    args = parser.parse_args()

    random.seed(7)
    torch.manual_seed(7)

    # CUDA_LAUNCH_BLOCKING = 1
    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_net = DQN(args).to(device)
    target_net = DQN(args).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    if args.optim == 'Adam':
        optimizer = optim.Adam(policy_net.parameters(), lr=args.lr)
    else:
        optimizer = optim.RMSprop(policy_net.parameters(), lr=args.lr)

    capacity = args.n_episodes * args.n_phones
    memory = ReplayMemory(capacity)

    Transition = namedtuple('Transition',
                            ('state', 'adj_c', 'phone', 'action', 'next_state', 'acc_reward'))

    stack_xa = StackXAction(maxsize=args.delay)
    stack_reward = StackReward(maxsize=args.delay)

    ################# random validation dataset #####################

    # Path to valid data
    # val_file_path = os.path.join(args.data_dir, 'val.h5')

    val_adj = torch.rand(args.val_batch, args.n_phones, args.n_phones)
    val_adj = (val_adj + val_adj.transpose(dim0=1, dim1=2)) / 2
    val_adj = (val_adj < args.rand_edge).type(torch.float32)
    for _j in range(args.n_phones):
        val_adj[:, _j, _j] = 1
    val_adj_c = (1 - val_adj)
    # Generate random allocation matrix for val dataset
    val_hot_idx = torch.randint(high=args.n_hosts, size=(args.val_batch * args.n_phones, ))  # host index
    val_x = F.one_hot(val_hot_idx, num_classes=args.n_hosts).reshape(
        16, args.n_phones, args.n_hosts).type(torch.float32)

    risk_init = torch.matmul(torch.matmul(val_x.transpose(dim0=1, dim1=2).to(device),
                                          val_adj_c.to(device)), val_x.to(device))
    risk_init = risk_init.diagonal(dim1=1, dim2=2).sum(dim=-1)

    #################################################################

    # Create log directory and save directory if it does not exist
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Create summary writer
    val_writer = SummaryWriter(
        log_dir=os.path.join(args.log_dir, "valid_{}".format(args.name_idx)))
    # Prepare checkpoint file
    checkpoint_file = os.path.join(args.save_dir, "checkpoint_{}.path".format(args.name_idx))

    n_phones = args.n_phones
    eps_end = args.eps_end
    eps_start = args.eps_start
    eps_decay = args.eps_decay
    delay = args.delay
    replay_batch = args.replay_batch
    gamma = args.gamma
    val_batch = args.val_batch

    # Initialize training
    steps_done = 0

    # Training loop
    for episode in tqdm(range(args.n_episodes)):
        # Generate random graph
        adj = torch.rand(1, args.n_phones, args.n_phones)
        adj = (adj + adj.transpose(dim0=1, dim1=2)) / 2
        adj = (adj < args.rand_edge).type(torch.float32)
        for _j in range(args.n_phones):
            adj[:, _j, _j] = 1
        adj_c = (1 - adj).to(device)
        # Generate random allocation matrix
        hot_idx = torch.randint(high=args.n_hosts, size=(1 * args.n_phones, ))  # host index
        x = F.one_hot(hot_idx, num_classes=args.n_hosts).reshape(
            1, args.n_phones, args.n_hosts).type(torch.float32).to(device)

        risk_bf = torch.matmul(torch.matmul(x.transpose(dim0=1, dim1=2), adj_c), x)
        risk_bf = risk_bf.diagonal(dim1=1, dim2=2).sum(dim=-1)

        for _i in range(n_phones):
            # select action
            sample = random.random()
            eps_threshold = eps_end + (eps_start - eps_end) * \
                            math.exp(-1. * steps_done / eps_decay)
            steps_done += 1
            if sample > eps_threshold:
                with torch.no_grad():
                    # t.max(1) will return largest column value of each row.
                    # second column on max result is index of where max element was
                    # found, so we pick action with the larger expected reward.
                    phone = torch.tensor([_i], device=device)
                    action = policy_net(x, adj_c, phone).max(1)[1].view(1, 1)
            else:
                action = \
                    torch.tensor([[random.randrange(args.n_hosts)]], device=device, dtype=torch.long)
            assert action.is_cuda
            stack_xa.put((x, action))

            # Compute the reward
            x_i = F.one_hot(action, num_classes=args.n_hosts).type(torch.float32).to(device)
            x[:, _i, :] = x_i.squeeze(dim=1)
            risk_af = torch.matmul(torch.matmul(x.transpose(dim0=1, dim1=2), adj_c), x)
            risk_af = risk_af.diagonal(dim1=1, dim2=2).sum(dim=-1)
            reward = (risk_bf - risk_af).item()
            assert risk_af.is_cuda
            assert risk_af.requires_grad is False
            stack_reward.put(reward)

            if _i+1 < n_phones:
                next_state = x
                risk_bf = risk_af
            else:
                next_state = None

            if _i+1 >= delay:
                state, action = stack_xa.get(0)
                assert state.is_cuda
                acc_reward = 0
                for _j in range(args.delay):
                    acc_reward += stack_reward.get(_j)
                acc_reward = torch.tensor([acc_reward], device=device)
                phone = torch.tensor([_i+1-delay], device=device)
                memory.push(state, adj_c, phone, action, next_state, acc_reward)

            # Optimize model
            if len(memory) >= replay_batch:
                transitions = memory.sample(replay_batch)
                # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
                # detailed explanation). This converts batch-array of Transitions
                # to Transition of batch-arrays.
                batch = Transition(*zip(*transitions))

                # Compute a mask of non-final states and concatenate the batch elements
                # (a final state would've been the one after which simulation ended)
                non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                        batch.next_state)), device=device, dtype=torch.bool)
                non_final_next_states = torch.cat([s for s in batch.next_state
                                                   if s is not None])
                non_final_next_adj_c = torch.cat([s for _j, s in enumerate(batch.adj_c)
                                                  if non_final_mask[_j].item() is True])
                non_final_next_phone = torch.cat([s for _j, s in enumerate(batch.phone)
                                                  if non_final_mask[_j].item() is True]) + delay
                assert len(non_final_next_states) == len(non_final_next_adj_c)
                assert non_final_next_adj_c.is_cuda
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
                # Expected values of actions for non_final_next_states are computed based
                # on the "older" target_net; selecting their best reward with max(1)[0].
                # This is merged based on the mask, such that we'll have either the expected
                # state value or 0 in case the state was final.
                next_state_values = torch.zeros(replay_batch, device=device)
                next_state_values[non_final_mask] = target_net(non_final_next_states,
                                                               non_final_next_adj_c ,
                                                               non_final_next_phone).max(1)[0].detach()

                # Compute the expected Q values
                assert next_state_values.shape == acc_reward_batch.shape
                expected_state_action_values = (next_state_values * gamma) + acc_reward_batch

                # Compute Huber loss
                loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                if args.clamp:
                    for param in policy_net.parameters():
                        param.grad.data.clamp_(-1, 1)
                optimizer.step()

        ##################### validation result #########################

        adj_c = val_adj_c.to(device)
        x = val_x.to(device)

        for _i in range(n_phones):
            with torch.no_grad():
                phone_set = torch.ones(val_batch).type(torch.int).to(device) * _i

                # select action
                action = policy_net(x, adj_c, phone_set).max(1)[1]

                # Compute the reward
                x_i = F.one_hot(action, num_classes=args.n_hosts).type(torch.float32).to(device)
                x[:, _i, :] = x_i
                assert x.is_cuda

        risk_final = torch.matmul(torch.matmul(x.transpose(dim0=1, dim1=2), adj_c), x)
        assert risk_final.diagonal(dim1=1, dim2=2).size() == (args.val_batch, args.n_hosts)
        assert risk_final.requires_grad is False
        risk_final = risk_final.diagonal(dim1=1, dim2=2).sum(dim=-1)

        ret = (risk_init - risk_final).mean().item()  # return per episode

        #################################################################

        # Monitor validation results every episode
        val_writer.add_scalar("data/return", ret, global_step=episode)

        # Update the target network, copying all weights and biases in DQN
        if episode % args.target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

    torch.save({"model": policy_net.state_dict()}, checkpoint_file)


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.Transition = namedtuple('Transition',
                                     ('state', 'adj_c', 'phone', 'action', 'next_state', 'acc_reward'))

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


class SkipConnection(nn.Module):

    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input):
        return input + self.module(input)


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


class BasicBlock(nn.Sequential):
    def __init__(self, args):
        super(BasicBlock, self).__init__(
            Normalization(args.embed_dim),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(args.embed_dim, args.enc_ff_hidden),
                    nn.ReLU(),
                    nn.Linear(args.enc_ff_hidden, args.embed_dim)
                ) if args.enc_ff_hidden > 0 else nn.Linear(args.embed_dim, args.embed_dim)
            ),
            Normalization(args.embed_dim)
        )


class MultiHeadAttention(nn.Module):
    def __init__(self, args, key_dim=None):
        super(MultiHeadAttention, self).__init__()

        val_dim = args.embed_dim // args.n_heads
        if key_dim is None:
            key_dim = val_dim

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.n_heads = args.n_heads
        self.val_dim = val_dim

        self.norm_factor = 1 / math.sqrt(key_dim)

        self.W_query = nn.Parameter(torch.Tensor(args.n_heads, args.embed_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(args.n_heads, args.embed_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(args.n_heads, args.embed_dim, val_dim))

        self.W_out = nn.Parameter(torch.Tensor(args.n_heads, val_dim, args.embed_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, h, adj_c):
        """
        :param h: data (batch, graph_size, embed_dim)
        """

        # h should be (1, graph_size, embed_dim)
        batch, graph_size, embed_dim = h.size()

        hflat = h.contiguous().view(-1, embed_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch, graph_size, -1)

        Q = torch.matmul(hflat, self.W_query).view(shp)  # (n_heads, 1, graph_size, key_dim)
        K = torch.matmul(hflat, self.W_key).view(shp)  # (n_heads, 1, graph_size, key_dim)
        V = torch.matmul(hflat, self.W_val).view(shp)  # (n_heads, 1, graph_size, val_dim)

        # Calculate compatibility (n_heads, 1, graph_size, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        adj_c = adj_c.unsqueeze(dim=0).expand(self.n_heads, *adj_c.size())
        assert compatibility.size() == adj_c.size()
        compatibility = \
            torch.where(adj_c == 0, -torch.tensor([float("inf")], device=self.device), compatibility)
        attn = torch.softmax(compatibility, dim=-1)
        attn = attn * adj_c

        heads = torch.matmul(attn, V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, embed_dim)
        ).view(batch, graph_size, embed_dim)

        return out + h


class GraphAttentionEncoder(nn.Module):
    def __init__(self, args):
        super(GraphAttentionEncoder, self).__init__()

        self.n_layers = args.n_layers
        self.init_embed = nn.Linear(args.n_hosts, args.embed_dim)

        assert args.embed_dim % args.n_heads == 0
        for _l in range(args.n_layers):
            setattr(self, "mhasc_{}".format(_l), MultiHeadAttention(args))
            setattr(self, "bb_{}".format(_l), BasicBlock(args))

    def forward(self, x, adj_c):
        """
        :param x: (batch, graph_size, node_dim) input node features
        """
        h = self.init_embed(x)
        for _l in range(self.n_layers):
            h = getattr(self, "mhasc_{}".format(_l))(h, adj_c)
            h = getattr(self, "bb_{}".format(_l))(h)

        return h


class MultiHeadAggregation(nn.Module):
    def __init__(self, args, key_dim=None):
        super(MultiHeadAggregation, self).__init__()

        val_dim = args.embed_dim // args.n_heads
        if key_dim is None:
            key_dim = val_dim

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.clip = args.clip

        self.n_heads = args.n_heads
        self.val_dim = val_dim

        self.norm_factor = 1 / math.sqrt(key_dim)

        self.W_query_bf = nn.Parameter(torch.Tensor(args.embed_dim, key_dim))
        self.W_key_bf = nn.Parameter(torch.Tensor(args.embed_dim, key_dim))
        self.W_val_bf = nn.Parameter(torch.Tensor(args.embed_dim, val_dim))
        self.W_out_bf = nn.Parameter(torch.Tensor(val_dim, args.embed_dim))

        self.W_query_af = nn.Parameter(torch.Tensor(args.embed_dim, key_dim))
        self.W_key_af = nn.Parameter(torch.Tensor(args.embed_dim, key_dim))
        self.W_val_af = nn.Parameter(torch.Tensor(args.embed_dim, val_dim))
        self.W_out_af = nn.Parameter(torch.Tensor(val_dim, args.embed_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

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

            h_i = h[_i, :, :].unsqueeze(dim=0)  # (1, graph_size, embed_dim)
            hiflat = h_i.contiguous().view(-1, embed_dim)  # (graph_size, embed_dim)
            h_ii = h_i[:, phone, :].unsqueeze(dim=1)  # (1, 1, embed_dim)
            hiiflat = h_ii.contiguous().view(-1, embed_dim)  # (1, embed_dim)
            assert hiiflat.is_cuda

            # last dimension can be different for keys and values
            shpi = (graph_size, -1)
            # last dimension is key_size
            shpii = (1, -1)

            ################## non-adjacent phones before ###############

            Q_bf = torch.mm(hiiflat, self.W_query_bf).view(shpii)  # queries (1, key_dim)
            K_bf = torch.mm(hiflat, self.W_key_bf).view(shpi)  # keys (graph_size, key_dim)
            V_bf = torch.mm(hiflat, self.W_val_bf).view(shpi)  # values (graph_size, val_dim)

            # Calculate compatibility before (1, phone)
            if self.clip:
                compatibility_bf = \
                    10 * torch.tanh(self.norm_factor * torch.mm(Q_bf, K_bf.T)[:, 0:phone])
            else:
                compatibility_bf = self.norm_factor * torch.mm(Q_bf, K_bf.T)[:, 0:phone]
            adj_c_bf = adj_c[_i, phone, 0:phone].unsqueeze(dim=0)
            assert compatibility_bf.size() == adj_c_bf.size()

            if adj_c_bf.sum().item() > 0:
                compatibility_bf = \
                    torch.where(adj_c_bf == 0, -torch.tensor([float("inf")], device=self.device), compatibility_bf)
                attn_bf = torch.softmax(compatibility_bf, dim=-1)
                attn_bf = attn_bf * adj_c_bf

                head_bf = torch.mm(attn_bf, V_bf[0:phone, :])  # (1, val_dim)

                g_bf = torch.mm(head_bf, self.W_out_bf)  # (1, embed_dim)
            else:
                g_bf = torch.zeros(1, embed_dim).to(self.device)

            g_bf_list.append(g_bf)

            ################## non-adjacent phones after ###############

            Q_af = torch.mm(hiiflat, self.W_query_af).view(shpii)  # queries (1, key_dim)
            K_af = torch.mm(hiflat, self.W_key_af).view(shpi)  # keys (graph_size, key_dim)
            V_af = torch.mm(hiflat, self.W_val_af).view(shpi)  # values (1, graph_size, val_dim)

            # Calculate compatibility before (1, graph_size-phone)
            if self.clip:
                compatibility_af = \
                    10 * torch.tanh(self.norm_factor * torch.matmul(Q_af, K_af.T)[:, phone:graph_size])
            else:
                compatibility_af = self.norm_factor * torch.matmul(Q_af, K_af.T)[:, phone:graph_size]
            adj_c_af = adj_c[_i, phone, phone:graph_size].unsqueeze(dim=0)
            assert compatibility_af.size() == adj_c_af.size()

            if adj_c_af.sum().item() > 0:
                compatibility_af = \
                    torch.where(adj_c_af == 0, -torch.tensor([float("inf")], device=self.device), compatibility_af)
                attn_af = torch.softmax(compatibility_af, dim=-1)
                attn_af = attn_af * adj_c_af

                head_af = torch.matmul(attn_af, V_af[phone:graph_size, :])  # (1, val_dim)

                g_af = torch.mm(head_af, self.W_out_af)  # (1, embed_dim)
            else:
                g_af = torch.zeros(1, embed_dim).to(self.device)

            g_af_list.append(g_af)

        return torch.cat(g_bf_list, dim=0), torch.cat(g_af_list, dim=0)


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()

        assert args.embed_dim % args.n_heads == 0
        self.mhg = MultiHeadAggregation(args)

        self.action = nn.Sequential(
            nn.Linear(2 * args.embed_dim, args.dec_ff_hidden),
            nn.ReLU(),
            nn.Linear(args.dec_ff_hidden, args.n_hosts))

    def forward(self, h, adj_c, phone_set):
        """
        :param h: (1, graph_size, embed_dim) embedding vector at hand
        :param phone_set: the set of phone indexes
        """
        g_bf, g_af = self.mhg(h, adj_c, phone_set)
        assert g_bf.is_cuda
        a = self.action(torch.cat((g_bf, g_af), dim=-1))  # action_values

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
    train()
