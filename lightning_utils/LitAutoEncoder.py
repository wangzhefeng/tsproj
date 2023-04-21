# -*- coding: utf-8 -*-

# ***************************************************
# * File        : LitAutoEncoder.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-21
# * Version     : 0.1.042115
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import transforms
import lightning.pytorch as pl

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# data
dataset = None
train_dataloader = DataLoader(
    dataset, 
    batch_size = 64,
    shuffle = True,
)
val_dataloader = DataLoader(
    dataset,
    batch_size = 64,
    shuffle = False,
)

# model
encoder = nn.Sequential(
    nn.Linear(28 * 28, 64),
    nn.ReLU(),
    nn.Linear(64, 3),
)
decoder = nn.Sequential(
    nn.Linear(3, 64),
    nn.ReLU(),
    nn.Linear(64, 28 * 28),
)

class LitModel(pl.LightningModule):

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self):
        embedding = self.encoder(x)
        return embedding
    
    def backward(self, loss, optimizer, optimizer_idx):
        loss.backward(retain_graph = True)

    def training_step(self, train_batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters, lr = 1e-3)
        return optimizer


# model
model = LitModel(encoder, decoder)
callbacks = [ConfusedLogitCallback(), StochasticWeightAveraging()]
trainer = pl.Trainer(
    accelerator = "auto", 
    strategy = "auto",
    device = "auto",
    precision = 16,
    limit_train_batches = 100, 
    min_epochs = 5,
    max_epochs = 10,
    callbacks = callbacks,
)
trainer.fit(
    model = model,
    train_dataloaders = train_dataloader,
    val_dataloaders = val_dataloader,
    datamodule = None,
    ckpt_path = None,
)

# ------------------------------
# load checkpoint
# ------------------------------
# TODO checkpoints callback
class Checkpoints(pl.Callback):

    def on_train_start(self, trainer, pl_module):
        aws.create_folder("checkpoints")

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        aws.save_checkpoint(decoder.weights, "tmp")

    def on_train_end(self, trainer, pl_module):
        aws.save_checkpoint(decoder.weights, "final")


checkpoint = "./lightning_logs/version_0/checkpoints/epoch=0-setp=100.ckpt"
model = LitAutoEncoder.load_from_checkpoint(checkpoint, encoder = encoder, decoder = decoder)

encoder = model.encoder
encoder.eval()




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
