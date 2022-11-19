'''
Baseline training and testing agent for the structure of the project
'''

import os
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR
from torch.utils.data import DataLoader


from tools.utils import prepare_device

from data_loaders.base_dataset import BaseDataset, BaseTestDataset
from models.baseline_models import BaseModel


class BaseAgent(ABC):
    """ Base class for all agents """
    def __init__(self, cfg, test=False, sweep=False):
        self.cfg = cfg
        self.is_test = test
        self.checkpoint_dir = cfg.model_dir

        self.batch_size = cfg.batch_size

        if self.cfg.use_gpu:
            self.device, self.device_ids = prepare_device(self.cfg.n_gpu)
        else:
            self.device, self.device_ids = prepare_device(0)

        self.model = None
        self.optimizer = None
        self.best_loss = float("inf")

    def build_model(self):
        self.model = BaseModel(self.cfg)

    def load_data(self):
        self.train_dataset = BaseDataset(data_dir=self.cfg.train_filenames)
        self.val_dataset = BaseDataset(data_dir=self.cfg.val_filenames)

        self.train_steps = len(self.train_dataset) // self.batch_size
        self.val_steps = len(self.val_dataset) // self.batch_size

        self.train_data_loader = DataLoader(self.train_dataset, batch_size=self.batch_size,\
                                            shuffle=True, num_workers=8, pin_memory=True)
        self.val_data_loader = DataLoader(self.val_dataset, batch_size=self.batch_size,\
                                            shuffle=False, num_workers=8, pin_memory=True)

    def train(self):
        self.best_loss = float("inf")
        val_loss_f = []
        train_loss_f = []

        self.build_model()
        self.criterion = self.build_loss_function()
        self.optimizer = self.build_optimizer()

        self.load_data()

        last_epoch = 0
        if os.path.exists(self.checkpoint_dir):
            last_epoch = self.load_model()

        self.scheduler = self.lr_scheduler(last_epoch)

        epochs = self.cfg.epochs
        self.train_writer = SummaryWriter(self.cfg.train_sum, "Train")
        self.val_writer = SummaryWriter(self.cfg.val_sum, "Val")

        for epoch in range(last_epoch + 1, epochs + 1):            
            _, msg = self.train_one_epoch(epoch)
            train_loss_f.append(msg)
            loss, msg = self.val_one_epoch(epoch)
            val_loss_f.append(msg)
            if loss < self.best_loss:
                self.best_loss = loss
                self.save_model(epoch)

            self.scheduler.step()

        with open(self.cfg.logs_dir + "train_loss.txt", "w+") as f:
            for msg in train_loss_f:
                f.write(msg + "\n")
        with open(self.cfg.logs_dir + "val_loss.txt", "w+") as f:
            for msg in val_loss_f:
                f.write(msg + "\n")

        self.train_writer.close()
        self.val_writer.close()

    def run_batch(self):
        pass

    def train_one_epoch(self):
        pass

    def val_one_epoch(self):
        pass

    def test(self):
        pass

    def lr_scheduler(self, last_epoch):
        scheduler = self.cfg.lr_scheduler.used
        if scheduler == "ExponentialLR":
            return ExponentialLR(optimizer=self.optimizer, gamma=self.cfg.lr_scheduler.ExponentialLR.decay)
        elif scheduler == "MultiStepLR":
            milestones = list(range(0, self.cfg.epochs, self.cfg.lr_scheduler.MultiStepLR.range))
            return MultiStepLR(self.optimizer, milestones=milestones, 
                            gamma=self.cfg.lr_scheduler.MultiStepLR.decay, last_epoch=last_epoch-1)

    def build_optimizer(self):
        optimizer = self.cfg.optimizer.used.lower()
        if optimizer == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=self.cfg.optimizer.Adam.lr, 
                                    weight_decay=self.cfg.optimizer.Adam.weight_decay)
        elif optimizer == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=self.cfg.optimizer.SGD.lr,
                                   weight_decay=self.cfg.optimizer.SGD.weight_decay)
        elif optimizer == "amsgrad":
            return torch.optim.Adam(self.model.parameters(), lr=self.cfg.optimizer.AmsGrad.lr,
                                    weight_decay=self.cfg.optimizer.AmsGrad.weight_decay, amsgrad=True)

    def build_loss_function(self):
        pass

    def save_model(self, epoch):
        ckpt = {'model': self.model.state_dict(),
                'optimizer':self.optimizer.state_dict(),
                'best_loss': self.best_loss,
                "epoch": epoch}
        torch.save(ckpt, self.checkpoint_dir)

    def load_model(self):
        ckpt = torch.load(self.checkpoint_dir)
        self.model.load_state_dict(ckpt['model'])
        if not self.is_test:
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.best_loss = ckpt['best_loss']

        return ckpt['epoch']

    def write_summary(self, summary_writer, total_loss, epoch):
        summary_writer.add_scalar('Loss', total_loss, epoch)

    def default_cfg(self):
        return {
            "lr": self.cfg.optimizer.AmsGrad.lr,
            "decay": self.cfg.lr_scheduler.ExponentialLR.decay
        }
