# -*- coding: utf-8 -*-
import os


root_dir = "/Users/zfwang/project/machinelearning"
project_dir = os.path.join(root_dir, "computer_vision")
data_dir = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"
# data_dir = os.path.join(project_dir, "timeseries_classification/FordA")
train_data_dir = os.path.join(data_dir, "FordA_TRAIN.tsv")
test_data_dir = os.path.join(data_dir, "FordA_TEST.tsv")
