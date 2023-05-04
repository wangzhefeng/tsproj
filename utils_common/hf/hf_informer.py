# -*- coding: utf-8 -*-


# ***************************************************
# * File        : hf_informer.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-18
# * Version     : 0.1.041821
# * Description : description
# * Link        : https://huggingface.co/docs/transformers/main/en/model_doc/informer#transformers.InformerModel
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

import torch
from huggingface_hub import hf_hub_download
from transformers import (
    InformerConfig, 
    InformerModel, 
    InformerForPrediction,
    InformerPreTrainedModel,
    INFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,
    INFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
)

import warnings
warnings.filterwarnings("ignore")


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------
# config
# ------------------------------
config = InformerConfig(prediction_length = 12)

# ------------------------------
# load pre-trained model
# ------------------------------
file = hf_hub_download(
    repo_id = "kashif/tourism-monthly-batch", 
    filename = "train-batch.pt", 
    repo_type = "dataset",
)
batch = torch.load(file)
print(f"\nbatch:\n{batch.keys()}")

# ------------------------------
# model
# ------------------------------
# model = InformerModel.from_pretrained("huggingface/informer-tourism-monthly")
model = InformerForPrediction.from_pretrained("huggingface/informer-tourism-monthly")
# view config
configuration = model.config
print(f"\nconfiguration:\n{configuration}")

# model training
outputs = model(
    past_values = batch["past_values"],
    past_time_features = batch["past_time_features"],
    past_observed_mask = batch["past_observed_mask"],
    static_categorical_features = batch["static_categorical_features"],
    # static_real_features=batch["static_real_features"],
    future_values = batch["future_values"],
    future_time_features = batch["future_time_features"],
)


# loss and backward
loss = outputs.loss
loss.backward()

# ------------------------------
# model inference
# ------------------------------
outputs = model.generate(
    past_values = batch["past_values"],
    past_time_features = batch["past_time_features"],
    past_observed_mask = batch["past_observed_mask"],
    static_categorical_features = batch["static_categorical_features"],
    # static_real_features = batch["static_real_features"],
    future_time_features = batch["future_time_features"],
)
mean_prediction = outputs.sequences.mean(dim = 1)
print(f"\nmean_prediciton\n{mean_prediction}")
print(mean_prediction.shape)



# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
