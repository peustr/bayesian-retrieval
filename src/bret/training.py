import logging
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

logger = logging.getLogger(__name__)


class DPRTrainer:
    def __init__(self, model, training_data, device):
        self.model = model
        self.training_data = training_data
        self.device = device

    def train(self, num_epochs=4, lr=5e-5, gamma=0.5, ckpt_file_name=None):
        optimizer = Adam(self.model.parameters(), lr=lr)
        scheduler = ExponentialLR(optimizer, gamma=gamma)
        if ckpt_file_name is not None:
            min_training_loss = 1e5
        else:
            min_training_loss = -1.0
        for epoch in range(1, num_epochs + 1):
            t_start = time.time()
            self.model.train()
            training_losses = []
            for qry, psg in self.training_data:
                qry = qry.to(self.device)
                psg = psg.to(self.device)
                optimizer.zero_grad()
                qry_reps, psg_reps = self.model(qry, psg)
                scores = qry_reps @ psg_reps.T
                targets = torch.arange(scores.size(0), device=self.device, dtype=torch.long) * (
                    psg_reps.size(0) // qry_reps.size(0)
                )
                loss = F.cross_entropy(scores, targets)
                loss.backward()
                optimizer.step()
                training_losses.append(loss.item())
            scheduler.step()
            avg_training_loss = np.mean(training_losses)
            t_end = time.time()
            logger.info("Epoch %d finished in %.2f minutes.", epoch, (t_end - t_start) / 60)
            logger.info("Average training loss: %.2f", avg_training_loss)
            if avg_training_loss < min_training_loss:
                torch.save(self.model.state_dict(), ckpt_file_name)
                min_training_loss = avg_training_loss
                logger.info("Model saved in: %s", ckpt_file_name)


class BayesianDPRTrainer(DPRTrainer):
    def __init__(self, model, training_data, device):
        super().__init__(model, training_data, device)

    def train(self, num_epochs=4, lr=5e-5, gamma=0.5, ckpt_file_name=None):
        optimizer = Adam(self.model.parameters(), lr=lr)
        scheduler = ExponentialLR(optimizer, gamma=gamma)
        if ckpt_file_name is not None:
            min_training_loss = 1e5
        else:
            min_training_loss = -1.0
        for epoch in range(1, num_epochs + 1):
            t_start = time.time()
            self.model.train()
            training_losses_ce = []
            training_losses_kld = []
            for qry, psg in self.training_data:
                qry = qry.to(self.device)
                psg = psg.to(self.device)
                optimizer.zero_grad()
                qry_reps, psg_reps = self.model(qry, psg)
                scores = qry_reps.mean(dim=1) @ psg_reps.mean(dim=1).T  # Averaging across num_samples.
                targets = torch.arange(scores.size(0), device=self.device, dtype=torch.long) * (
                    psg_reps.size(0) // qry_reps.size(0)
                )
                loss_ce = F.cross_entropy(scores, targets)
                loss_kld = self.model.kl() / len(self.training_data.dataset)
                loss = loss_ce + loss_kld
                loss.backward()
                optimizer.step()
                training_losses_ce.append(loss_ce.item())
                training_losses_kld.append(loss_kld.item())
            scheduler.step()
            avg_training_loss_ce = np.mean(training_losses_ce)
            avg_training_loss_kld = np.mean(training_losses_kld)
            t_end = time.time()
            logger.info("Epoch %d finished in %.2f minutes.", epoch, (t_end - t_start) / 60)
            logger.info("Average training loss: %.2f", avg_training_loss_ce)
            logger.info("Average KL divergence: %.2f", avg_training_loss_kld)
            if avg_training_loss_ce < min_training_loss:
                torch.save(self.model.state_dict(), ckpt_file_name)
                min_training_loss = avg_training_loss_ce
                logger.info("Model saved in: %s", ckpt_file_name)
