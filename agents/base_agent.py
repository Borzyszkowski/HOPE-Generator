""" Baseline training and testing agent for the structure of the project """

import logging
import os
from abc import ABC
from datetime import datetime

import torch
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR
from torch.utils.data import DataLoader

from agents.training_tools.losses import nll_loss
from data_loaders.base_dataset import BaseDataset
from tools.tb_writer import TensorboardWriter
from tools.utils import prepare_device

from data_loaders.mnist_dataloader import MnistDataLoader
from models.mnist_model import MNISTModel


class BaseAgent(ABC):
    """ Base class for all agents """

    def __init__(self, cfg, test=False):
        self.cfg = cfg
        self.is_test = test
        self.checkpoint_dir = cfg.result_dir
        self.start_experiment_message()
        self.batch_size = cfg.batch_size

        if self.cfg.use_gpu:
            self.device, self.device_ids = prepare_device(self.cfg.n_gpu)
        else:
            self.device, self.device_ids = prepare_device(0)

        self.model = None
        self.optimizer = None
        self.best_loss = float("inf")
        self.writer = TensorboardWriter(self.cfg.log_dir, self.cfg.tensorboard)

    def start_experiment_message(self):
        """ Generate message about start of the experiment and configuration details """
        logging.info(f'[{self.cfg["run_id"]}] - HOPE Generator has started!')
        logging.info(f'log dir: {self.cfg.log_dir}')
        logging.info(f'checkpoint dir: {self.cfg.save_dir}')
        logging.info(f'Torch Version: {torch.__version__}')

    def build_model(self):
        """ Select the desired model architecture and initialize it """
        logging.info(f"Selected model name: {self.cfg.model.used}")
        # TODO: Add model selection and remove the MNISTModel
        if self.cfg.model.used == "MNISTModel":
            self.model = MNISTModel(self.cfg)

    def load_data(self):
        logging.info(f"Loading data: {self.cfg.train_filenames}")

        # TODO: Add proper data loader and remove the MNISTModel
        if self.cfg.model.used == "MNISTModel":
            self.train_data_loader = MnistDataLoader()
            self.val_data_loader = self.train_data_loader.split_validation()
            return

        self.train_dataset = BaseDataset(data_dir=self.cfg.train_filenames)
        self.val_dataset = BaseDataset(data_dir=self.cfg.val_filenames)

        self.train_steps = len(self.train_dataset) // self.batch_size
        self.val_steps = len(self.val_dataset) // self.batch_size

        self.train_data_loader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                            shuffle=True, num_workers=8, pin_memory=True)
        self.val_data_loader = DataLoader(self.val_dataset, batch_size=self.batch_size,
                                          shuffle=False, num_workers=8, pin_memory=True)

    def train(self):
        """ Run the main training logic """
        val_loss_f = []
        train_loss_f = []

        self.load_data()
        self.build_model()
        self.optimizer = self.build_optimizer()
        self.criterion = self.build_loss_function()

        last_epoch = 0
        if os.path.exists(self.cfg.load_model):
            last_epoch = self.load_model()
        self.scheduler = self.lr_scheduler(last_epoch)
        epoch_number = self.cfg.epochs

        # Run the main training loop
        start_time = datetime.now().replace(microsecond=0)
        logging.info(f"Started training at {start_time} for {epoch_number} epochs\n")

        for epoch in range(last_epoch + 1, epoch_number + 1):

            # Run the training epoch
            loss = self.train_one_epoch(epoch)
            train_loss_f.append(loss)

            # Run the validation epoch
            loss = self.val_one_epoch(epoch)
            val_loss_f.append(loss)

            # Save the best model
            if loss < self.best_loss:
                self.best_loss = loss
                self.save_model(epoch)
            self.scheduler.step()

        # TODO: save the loss results
        # self.save_loss_results(train_loss_f, "train_loss.txt")
        # self.save_loss_results(val_loss_f, "val_loss.txt")

        # finish the training
        end_time = datetime.now().replace(microsecond=0)
        logging.info(f'Finished Training at {end_time}')
        logging.info(f'Training done in {(end_time - start_time)}!')

    def train_one_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_data_loader):

            # copy data to the device
            data, target = data.to(self.device), target.to(self.device)

            # run backpropagation
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            # TODO: save metrics to the tensorboard

            # print the loss message
            if batch_idx % self.cfg.log_every_iteration == 0:
                self._iter_message('Train', epoch, batch_idx, loss, self.train_data_loader)

        # TODO: The function shouldn't return just the last loss
        return loss

    def val_one_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.val_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                # compute loss
                output = self.model(data)
                loss = self.criterion(output, target)

                # TODO: save metrics to the tensorboard

                # print the loss message
                if batch_idx % self.cfg.log_every_iteration == 0:
                    self._iter_message('Val', epoch, batch_idx, loss, self.val_data_loader)

        # TODO: The function shouldn't return just the last loss
        return loss

    def test(self):
        # TODO: Implement the test logic
        pass

    def lr_scheduler(self, last_epoch):
        """ Select and build the learning rate scheduler """
        scheduler = self.cfg.lr_scheduler.used
        logging.info(f'Using learning rate scheduler: {scheduler}')
        if scheduler == "ExponentialLR":
            return ExponentialLR(optimizer=self.optimizer, gamma=self.cfg.lr_scheduler.ExponentialLR.decay)
        elif scheduler == "MultiStepLR":
            milestones = list(range(0, self.cfg.epochs, self.cfg.lr_scheduler.MultiStepLR.range))
            return MultiStepLR(self.optimizer, milestones=milestones,
                               gamma=self.cfg.lr_scheduler.MultiStepLR.decay, last_epoch=last_epoch - 1)

    def build_optimizer(self):
        """ Select and build the optimizer """
        optimizer = self.cfg.optimizer.used.lower()
        logging.info(f'Using optimizer: {optimizer}')
        params = [var[1] for var in self.model.named_parameters()]
        params_number = sum(p.numel() for p in params if p.requires_grad)
        logging.info('Total trainable parameters of the model: %2.2fM' % (params_number * 1e-6))
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
        """ Select and build the loss funtion """
        # TODO: Implement selection of the loss functions
        logging.info(f'Using loss funtion: {"TODO"}')
        return nll_loss

    def save_loss_results(self, loss, filename):
        """ Save loss values to the text file """
        with open(os.path.join(self.cfg.save_dir, filename), "w+") as f:
            for msg in loss:
                f.write(msg + "\n")

    def save_model(self, epoch):
        """ Save the model checkpoint """
        ckpt = {'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_loss': self.best_loss,
                "epoch": epoch}
        checkpoint_path = os.path.join(self.cfg.save_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(ckpt, checkpoint_path)
        logging.info(f"Saved checkpoint at: {checkpoint_path}")

    def load_model(self):
        """ Load model checkpoint from the specified destination """
        logging.info(f'Loading model checkpoint from: {self.cfg.load_model}')
        ckpt = torch.load(self.cfg.load_model)
        self.model.load_state_dict(ckpt['model'])
        if not self.is_test:
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.best_loss = ckpt['best_loss']
        return ckpt['epoch']

    # def write_summary(self, summary_writer, total_loss, epoch):
    #     summary_writer.add_scalar('Loss', total_loss, epoch)
    #
    # def default_cfg(self):
    #     return {
    #         "lr": self.cfg.optimizer.AmsGrad.lr,
    #         "decay": self.cfg.lr_scheduler.ExponentialLR.decay
    #     }

    def _iter_message(self, mode, epoch, batch_idx, loss, data_loader):
        """ Generate the message after each interation """
        progress = self._progress(batch_idx, data_loader)
        logging.info(f'{mode} Epoch: {epoch} Iter: {batch_idx} Progress: {progress} Loss: {loss.item():.6f}')

    def _progress(self, batch_idx, data_loader):
        """ Show the training progress of each epoch """
        base = '[{}/{} ({:.0f}%)]'
        current = batch_idx * data_loader.batch_size
        if hasattr(data_loader, 'n_samples'):
            total = data_loader.n_samples
        else:
            total = len(data_loader) * data_loader.batch_size
        return base.format(current, total, 100.0 * current / total)
