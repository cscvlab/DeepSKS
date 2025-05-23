import os, random, sys
import numpy as np
import torch
import torchvision
import torch.optim as optim
import torch.nn.functional as F
import math
import torch.nn as nn
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from kornia.geometry import get_perspective_transform, warp_perspective
import torchvision.transforms as transforms  

class Logger_(object):
    def __init__(self, filename=None, stream=sys.stdout):
            self.terminal = stream
            self.log = open(filename, 'a')

    def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

    def flush(self):
            pass


class Logger:
    def __init__(self, model, scheduler, args):
        self.model = model
        self.args = args
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss_dict = {}
        self.train_mace_list = []
        self.train_steps_list = []
        self.val_steps_list = []
        self.val_results_dict = {}

    def _print_training_status(self):
        metrics_data = [np.mean(self.running_loss_dict[k]) for k in sorted(self.running_loss_dict.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps + 1, self.scheduler.get_lr()[0])
        metrics_str = ("{:10.4f}, " * len(metrics_data[:-1])).format(*metrics_data[:-1])

        # Compute time left
        time_left_sec = (self.args.num_steps - (self.total_steps + 1)) * metrics_data[-1]
        time_left_sec = time_left_sec.astype(np.int_)
        time_left_hms = "{:02d}h{:02d}m{:02d}s".format(time_left_sec // 3600, time_left_sec % 3600 // 60,
                                                       time_left_sec % 3600 % 60)
        time_left_hms = f"{time_left_hms:>12}"
        # print the training status
        print(training_str + metrics_str + time_left_hms)

        # logging running loss to total loss
        self.train_mace_list.append(np.mean(self.running_loss_dict['mace']))
        self.train_steps_list.append(self.total_steps)

        for key in self.running_loss_dict:
            self.running_loss_dict[key] = []

    def push(self, metrics):
        self.total_steps += 1
        for key in metrics:
            if key not in self.running_loss_dict:
                self.running_loss_dict[key] = []

            self.running_loss_dict[key].append(metrics[key])

        if self.total_steps % self.args.print_freq == self.args.print_freq - 1:
            self._print_training_status()
            self.running_loss_dict = {}


def plot_train(logger, args):
    # plot training curve
    plt.figure()
    plt.plot(logger.train_steps_list, logger.train_mace_list)
    plt.xlabel('x_steps')
    plt.ylabel('EPE')
    plt.title('Running training error (EPE)')
    plt.savefig(args.output + "/train_epe.png", bbox_inches='tight')
    plt.close()


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=args.lr, total_steps=args.num_steps + 100,
                                              pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')
    return optimizer, scheduler
