# -*- coding: utf-8 -*-

# ***************************************************
# * File        : neural_network.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-12-13
# * Version     : 0.1.121323
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        # 2d cnn layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride = 1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        # dropout laery
        self.dropout1 = nn.Dropout(p = 0.5)
        self.dropout2 = nn.Dropout(p = 0.5)
        # fully connected layer
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.dropout2(x)

        return x




# 测试代码 main 函数
def main():
    # ------------------------------
    # Net
    # ------------------------------
    # net
    my_nn = Net()
    print(my_nn)

    # optimizer
    optimizer = optim.SGD(my_nn.parameters(), lr = 0.001, momentum=0.9)

    # criterizer
    loss = None
    # ------------------------------
    # state_dict
    # ------------------------------
    # model's state_dict
    print("Model's state_dict:")
    for param_tensor in my_nn.state_dict():
        print(param_tensor, "\t", my_nn.state_dict()[param_tensor].size())
    print()
    # optimizer's state_dict
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])

    # ------------------------------
    # data
    # ------------------------------
    data = None

    # ------------------------------
    # model
    # ------------------------------
    # training epoch
    epoch = None
    
    # model training 
    model = my_nn(data)

    # ------------------------------
    # saving & loading model for inference
    # ------------------------------
    # save/load state_dict
    # --------------------
    # model save
    torch.save(model.state_dict(), "./saved_models/my_nn.pt")
    torch.save(model.state_dict(), "./saved_models/my_nn.pth")
    # model load
    model.load_state_dict(torch.load("./saved_models/my_nn.pt"), weights_only = True)
    model.load_state_dict(torch.load("./saved_models/my_nn.pth"), weights_only = True)
    # set dropout and batch normalization layers to evaluation model before running inference 
    model.eval()

    # save/load entire model
    # ----------------------
    # save
    torch.save(model, "./saved_models/my_nn.pt")
    # load
    model = torch.load("./saved_models/my_nn.pt", weights_only=False)
    model.eval()
    
    # export/load model in TorchScript format
    # ---------------------
    # model save
    model_scripted = torch.jit.script(model)
    model_scripted.save("./saved_models/model_scripted.pt")
    # model load
    model = torch.jit.load("./saved_models/model_scripted.pt")
    model.eval()

    # ------------------------------
    # saving & Loading checkpoint for inference or resuming traing
    # ------------------------------
    # model save
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        "./checkpoints/",
    )
    # model load
    checkpoint = torch.load("./checkpoints/", weights_only=True)
    epoch = checkpoint["epoch"]
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    loss = checkpoint["loss"]
    
    model.eval()
    # or
    model.train()

    # ------------------------------
    # saving multiple models in one file
    # ------------------------------

    # ------------------------------
    # warmstarting model using paramters from different model
    # ------------------------------

    # ------------------------------
    # saving & loading model accross device
    # ------------------------------

    

if __name__ == "__main__":
    main()
