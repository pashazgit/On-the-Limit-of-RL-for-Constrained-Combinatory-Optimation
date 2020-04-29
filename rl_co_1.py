
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
                    default="/home/pasha/scratch/datasets/graphs",
                    help="")

parser.add_argument("--save_dir", type=str,
                    default="/home/pasha/scratch/jobs/output/RL-CO/saves",
                    help="Directory to save the best model")

parser.add_argument("--log_dir", type=str,
                    default="/home/pasha/scratch/jobs/output/RL-CO/logs",
                    help="Directory to save logs and current model")

parser.add_argument("--lr", type=float,
                    default=1e-4,
                    help="Learning rate (gradient step size)")

parser.add_argument("--scale", type=float,
                    default=2000,
                    help="")

parser.add_argument("--batch_size", type=int,
                    default=512,
                    help="Size of each training batch")

parser.add_argument("--p", type=int,
                    default=64,
                    help="")

parser.add_argument("--T", type=int,
                    default=4,
                    help="")

parser.add_argument("--phones", type=int,
                    default=200,
                    help="")

parser.add_argument("--hosts", type=int,
                    default=5,
                    help="")

parser.add_argument("--num_epoch", type=int,
                    default=40,
                    help="Number of epochs to train")

parser.add_argument("--val_intv", type=int,
                    default=500,
                    help="Validation interval")

parser.add_argument("--rep_intv", type=int,
                    default=500,
                    help="Report interval")

# parser.add_argument("--l2_reg", type=float,
#                     default=1e-4,
#                     help="L2 Regularization strength")

parser.add_argument("--resume", type=str2bool,
                    default=True,
                    help="Whether to resume training from existing checkpoint")

parser.add_argument("--name_idx", type=int,
                    default=0,
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

    # Initialize datasets for both training and validation
    tr_file_path = os.path.join(args.data_dir, 'tr.h5')
    val_file_path = os.path.join(args.data_dir, 'val.h5')
    tr_data = graphs(tr_file_path)
    val_data = graphs(val_file_path)

    # Create data loader for training and validation.
    tr_loader = DataLoader(
        dataset=tr_data,
        batch_size=args.batch_size,
        num_workers=2,
        shuffle=True)
    val_loader = DataLoader(
        dataset=val_data,
        batch_size=100,
        num_workers=2,
        shuffle=False)

    # Create model instance
    model = model_class(tr_data.al_shp, args.p)
    print('\nmodel created')
    # Move model to gpu if cuda is available
    if torch.cuda.is_available():
        # model = torch.nn.DataParallel(model).cuda()
        model = model.cuda()
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
    iter_idx = -1  # make counter start at zero
    best_val_ret = 0  # to check if best validation accuracy
    # Prepare checkpoint file and model file to save and load from
    checkpoint_file = os.path.join(args.save_dir, "checkpoint_{}.path".format(args.name_idx))
    bestmodel_file = os.path.join(args.save_dir, "bestmodel_{}.path".format(args.name_idx))

    # Check for existing training results. If it exists, and the configuration
    # is set to resume `args.resume==True`, resume from previous training. If
    # not, delete existing checkpoint.
    if os.path.exists(checkpoint_file):
        if args.resume:
            print("Checkpoint found! Resuming")
            # Read checkpoint file.
            load_res = torch.load(
                checkpoint_file,
                map_location="cpu")
            start_epoch = load_res["epoch"]
            # Resume iterations
            iter_idx = load_res["iter_idx"]
            # Resume best va result
            best_val_ret = load_res["best_val_ret"]
            # Resume model
            model.load_state_dict(load_res["model"])
            # Resume optimizer
            optimizer.load_state_dict(load_res["optimizer"])
        else:
            os.remove(checkpoint_file)

    # Training loop
    for epoch in tqdm(range(start_epoch, args.num_epoch)):

        for adj, al in tr_loader:
            iter_idx += 1  # Counter
            # Send data to GPU if we have one
            if torch.cuda.is_available():
                adj, al = adj.cuda(), al.cuda()
            adj_c = 1 - adj
            assert adj_c.is_cuda
            assert al.is_cuda
            # Apply the model to obtain scores (forward pass)
            logits = model(al, adj_c, args.T)
            log_prob = F.log_softmax(logits, dim=-1).max(dim=-1)[0].sum(dim=-1, keepdim=True)
            # Compute the return
            t = logits.detach()  # remove the logits from propagation graph
            al_n = F.one_hot(t.argmax(dim=-1), num_classes=args.hosts).to(torch.float32).cuda()
            risk_n = torch.matmul(torch.matmul(al_n.transpose(dim0=1, dim1=2), adj_c), al_n)
            risk_n = risk_n.diagonal(dim1=1, dim2=2).sum(dim=-1, keepdim=True)
            risk = torch.matmul(torch.matmul(al.transpose(dim0=1, dim1=2), adj_c), al)
            risk = risk.diagonal(dim1=1, dim2=2).sum(dim=-1, keepdim=True)
            ret = (risk - risk_n) / args.scale
            # back-propagation
            reinforce = ret * log_prob
            loss = -reinforce.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Monitor results every report interval
            if iter_idx % args.rep_intv == 0:
                tr_writer.add_scalar("data/return", ret.mean(), global_step=iter_idx)
                # Save
                torch.save({
                    "epoch": epoch,
                    "iter_idx": iter_idx,
                    "best_val_ret": best_val_ret,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }, checkpoint_file)

            # Validate results every validation interval
            if iter_idx % args.val_intv == 0:
                # Set model for evaluation
                model.eval()

                val_ret = []  # List to contain all returns for all the val batches
                for adj, al in val_loader:
                    # Send data to GPU if we have one
                    if torch.cuda.is_available():
                        adj, al = adj.cuda(), al.cuda()
                    adj_c = 1 - adj
                    # Apply forward pass to compute the returns for each of the val batches
                    with torch.no_grad():
                        logits = model(al, adj_c, args.T)
                        # Compute return and store as numpy
                        t = logits
                        al_n = F.one_hot(t.argmax(dim=-1), num_classes=args.hosts).to(torch.float32).cuda()
                        risk_n = torch.matmul(torch.matmul(al_n.transpose(dim0=1, dim1=2), adj_c), al_n)
                        risk_n = risk_n.diagonal(dim1=1, dim2=2).sum(dim=-1, keepdim=True)
                        risk = torch.matmul(torch.matmul(al.transpose(dim0=1, dim1=2), adj_c), al)
                        risk = risk.diagonal(dim1=1, dim2=2).sum(dim=-1, keepdim=True)
                        ret = risk - risk_n
                        val_ret += [ret.cpu().numpy()]
                val_ret_avg = np.mean(val_ret)
                # Write return to tensorboard, using keywords `return`.
                val_writer.add_scalar("data/return", val_ret_avg, global_step=iter_idx)

                # Set model back for training
                model.train()

                if val_ret_avg > best_val_ret:
                    best_val_ret = val_ret_avg
                    # Save
                    torch.save({
                        "model": model.state_dict(),
                    }, bestmodel_file)


def test(args):
    pass


class graphs(Dataset):
    def __init__(self, hdf5_file_path):
        self.hdf5_file = h5py.File(hdf5_file_path, mode='r')
        self.transform = transforms.ToTensor()
        self.adj_shp = self.hdf5_file["adj"][0].shape
        self.al_shp = self.hdf5_file["al"][0].shape

    def __len__(self):
        return len(self.hdf5_file["adj"])

    def __getitem__(self, idx):
        adj = self.hdf5_file["adj"][idx]
        al = self.hdf5_file["al"][idx]
        adj = self.transform(adj).squeeze()
        al = self.transform(al).squeeze()
        return adj, al


class model_class(nn.Module):
    def __init__(self, al_shape, p):
        super(model_class, self).__init__()
        self.phones, self.hosts = al_shape
        self.p = p

        self.theta1 = nn.Linear(self.hosts, p, bias=False)
        self.pre_linear1 = nn.Linear(p, p)
        self.pre_linear2 = nn.Linear(p, p)
        self.theta2 = nn.Linear(p, p, bias=False)
        self.post_linear1 = nn.Linear(p, p)

        self.output = nn.Linear(p, self.hosts)

    def forward(self, al, adj_c, T):
        mu = F.relu(self.theta1(al.reshape(-1, self.hosts)))
        for t in range(T-1):
            mu = F.relu(self.pre_linear1(mu))
            mu = F.relu(self.pre_linear2(mu))
            mu = torch.matmul(adj_c, mu.reshape(-1, self.phones, self.p))
            mu = self.theta2(mu.reshape(-1, self.p))
            mu = F.relu(self.post_linear1(mu))
            # mu = F.relu(self.theta1(al.reshape(-1, self.hosts)) + mu)

        mu = self.output(mu)

        return mu.reshape(-1, self.phones, self.hosts)


# def model_loss(args, model):
#     loss = 0
#     for name, param in model.named_parameters():
#         if "weight" in name:
#             loss += torch.sum(param**2)
#     return loss * args.l2_reg


if __name__ == "__main__":
    main()
