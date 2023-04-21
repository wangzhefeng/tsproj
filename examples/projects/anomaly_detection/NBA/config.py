# -*- coding: utf-8 -*-
import os


root_dir = "/Users/zfwang/project/machinelearning"
project_dir = os.path.join(root_dir, "computer_vision")
data_dir = "https://raw.githubusercontent.com/numenta/NAB/master/data/"
train_data_dir = os.path.join(data_dir, "artificialNoAnomaly/art_daily_small_noise.csv")
test_data_dir = os.path.join(data_dir, "artificialWithAnomaly/art_daily_jumpsup.csv")
