# -*- coding: utf-8 -*-

# ***************************************************
# * File        : test.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2025-01-19
# * Version     : 0.1.011915
# * Description : https://github.com/huggingface/blog/blob/main/time-series-transformers.md
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from typing import Optional, Iterable

from gluonts.time_feature import (
    get_lags_for_frequency,
    time_features_from_frequency_str,
    TimeFeature,
)
from gluonts.dataset.field_names import FieldName
from gluonts.transform import (
    AddAgeFeature,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    RemoveFields,
    SelectFields,
    SetField,
    TestSplitSampler,
    Transformation,
    ValidationSplitSampler,
    VstackFeatures,
    RenameFields,
)
from gluonts.transform.sampler import InstanceSampler
from gluonts.itertools import Cached, Cyclic
from gluonts.dataset.loader import as_stacked_batches
from transformers import (
    TimeSeriesTransformerConfig, 
    TimeSeriesTransformerForPrediction,
    PretrainedConfig,
)
import torch
from torch.optim import AdamW
from accelerate import Accelerator

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# ------------------------------
# params
# ------------------------------
freq = "1m"
prediction_length = 24

# ------------------------------
# 数据
# ------------------------------
train_dataset = None
test_dataset = None

train_sampler = None
validation_sampler = None

# ------------------------------
# 定义模型
# ------------------------------
config = TimeSeriesTransformerConfig(
    prediction_length=prediction_length,
    context_length=prediction_length * 2,
    lags_sequence=get_lags_for_frequency(freq),  # lags
    num_time_features=len(time_features_from_frequency_str(freq)) + 1,  # 2 time features(month of year, age)
    num_static_categorical_features=1,  # time series ID
    cardinality=[len(train_dataset)],  # length of values
    embedding_dimension=[2],  # embedding of size 2
    encoder_layers=4,  # transformer params
    decoder_layers=4,
    d_model=128,
)

model = TimeSeriesTransformerForPrediction(config)
print(model.config.distribution_output)


def create_transformation(freq: str, config: PretrainedConfig) -> Transformation:
    remove_field_names = []
    if config.num_static_real_features == 0:
        remove_field_names.append(FieldName.FEAT_STATIC_REAL)
    if config.num_dynamic_real_features == 0:
        remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)
    if config.num_static_categorical_features == 0:
        remove_field_names.append(FieldName.FEAT_STATIC_CAT)
    
    return Chain(
        # step 1: remove static/dynamic fields if not specified
        [RemoveFields(field_names=remove_field_names)]
        # step 2: convert the data to Numpy(potentially not needed)
        + (
            [
                AsNumpyArray(
                    field = FieldName.FEAT_STATIC_CAT,
                    expected_ndim = 1,
                    dtype = int,
                )
            ]
            if config.num_static_categorical_features > 0 
            else []
        )
        + (
            [
                AsNumpyArray(
                    field = FieldName.FEAT_STATIC_REAL,
                    expected_ndim = 1,
                )
            ]
            if config.num_static_real_features > 0
            else []
        )
        + [
            AsNumpyArray(
                field = FieldName.TARGET,
                expected_ndim = 1 if config.input_size == 1 else 2,
            ),
            # step 3: handle the NaN's by filling in the target with zero
            # and return the mask (which is in the observed values)
            # true for observed values, false for nan's
            # the decoder uses this mask (no loss is incurred for unobserved values)
            # see loss_weights inside the xxxForPrediction model# step 3: handle the NaN's by filling in the target with zero and return the mask
            AddObservedValuesIndicator(
                target_field = FieldName.TARGET,
                output_field = FieldName.OBSERVED_VALUES,
            ),
            # step 4: add temporal features based on freq of the dataset
            # month of year in the case when freq="M"
            # these serve as positional encodings
            AddTimeFeatures(
                start_field = FieldName.START,
                target_field = FieldName.TARGET,
                output_field = FieldName.FEAT_TIME,
                time_features = time_features_from_frequency_str(freq),
                pred_length = config.prediction_length,
            ),
            # step 5: add another temporal feature (just a single number)
            # tells the model where in its life the value of the time series is,
            # sort of a running counter
            AddAgeFeature(
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_AGE,
                pred_length=config.prediction_length,
                log_scale=True,
            ),
            # step 6: vertically stack all the temporal features into the key FEAT_TIME
            VstackFeatures(
                output_field=FieldName.FEAT_TIME,
                input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE]
                + (
                    [FieldName.FEAT_DYNAMIC_REAL]
                    if config.num_dynamic_real_features > 0
                    else []
                ),
            ),
            # step 7: rename to match HuggingFace names
            RenameFields(
                mapping={
                    FieldName.FEAT_STATIC_CAT: "static_categorical_features",
                    FieldName.FEAT_STATIC_REAL: "static_real_features",
                    FieldName.FEAT_TIME: "time_features",
                    FieldName.TARGET: "values",
                    FieldName.OBSERVED_VALUES: "observed_mask",
                }
            ),
        ]
    )


def create_instance_splitter(mode: str, 
                             cfg: TimeSeriesTransformerConfig, 
                             default_catchall_name: str = "target") -> InstanceSplitter:
    assert mode in ["training", "validation", "test"]

    instance_sampler = {
        "train": train_sampler or 
        ExpectedNumInstanceSampler(num_instances = 1.0, min_future = config.prediction_length),
        "validation": validation_sampler or 
        ValidationSplitSampler(min_future = config.prediction_length),
        "test": TestSplitSampler(),
    }[mode]
    
    return InstanceSplitter(
        target_field = "values",
        is_pad_field = FieldName.IS_PAD,
        start_field = FieldName.START,
        forecast_start_field = FieldName.FORECAST_START,
        instance_sampler = instance_sampler,
        past_length = config.context_length + max(config.lags_sequence),
        future_length = config.prediction_length,
        time_series_fields = ["time_feaures", "observed_mask"],
    )


def create_train_dataloader(config: PretrainedConfig, 
                            freq,
                            data,
                            batch_size: int,
                            num_batches_per_epoch: int,
                            shuffle_buffer_length: Optional[int] = None,
                            cache_data: bool = True,
                            **kwargs) -> Iterable:
    PREDICTION_INPUT_NAMES = [
        "past_time_features",
        "past_values",
        "past_observed_mask",
        "future_time_features",
    ]
    if config.num_static_categorical_features > 0:
        PREDICTION_INPUT_NAMES.append("static_categorical_features")
    if config.num_static_real_features > 0:
        PREDICTION_INPUT_NAMES.append("static_real_features")
    TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
        "future_values",
        "future_observed_mask",
    ]

    transformation = create_transformation(freq, config)
    transformed_data = transformation.apply(data, is_train = True)
    if cache_data:
        transformed_data = Cached(transformed_data)

    # we initialize a Training instance
    instance_splitter = create_instance_splitter(config, "train")

    # the instance splitter will sample a window of
    # context length + lags + prediction length (from the 366 possible transformed time series)
    # randomly from within the target time series and return an iterator.
    stream = Cyclic(transformed_data).stream()
    training_instances = instance_splitter.apply(stream)
    
    return as_stacked_batches(
        training_instances,
        batch_size=batch_size,
        shuffle_buffer_length=shuffle_buffer_length,
        field_names=TRAINING_INPUT_NAMES,
        output_type=torch.tensor,
        num_batches_per_epoch=num_batches_per_epoch,
    )


def create_backtest_dataloader(config: PretrainedConfig,
                               freq,
                               data,
                               batch_size: int,
                               **kwargs):
    PREDICTION_INPUT_NAMES = [
        "past_time_features",
        "past_values",
        "past_observed_mask",
        "future_time_features",
    ]
    if config.num_static_categorical_features > 0:
        PREDICTION_INPUT_NAMES.append("static_categorical_features")

    if config.num_static_real_features > 0:
        PREDICTION_INPUT_NAMES.append("static_real_features")

    transformation = create_transformation(freq, config)
    transformed_data = transformation.apply(data)

    # we create a Validation Instance splitter which will sample the very last
    # context window seen during training only for the encoder.
    instance_sampler = create_instance_splitter(config, "validation")

    # we apply the transformations in train mode
    testing_instances = instance_sampler.apply(transformed_data, is_train=True)
    
    return as_stacked_batches(
        testing_instances,
        batch_size=batch_size,
        output_type=torch.tensor,
        field_names=PREDICTION_INPUT_NAMES,
    )


def create_test_dataloader(config: PretrainedConfig,
                           freq,
                           data,
                           batch_size: int,
                           **kwargs):
    PREDICTION_INPUT_NAMES = [
        "past_time_features",
        "past_values",
        "past_observed_mask",
        "future_time_features",
    ]
    if config.num_static_categorical_features > 0:
        PREDICTION_INPUT_NAMES.append("static_categorical_features")

    if config.num_static_real_features > 0:
        PREDICTION_INPUT_NAMES.append("static_real_features")

    transformation = create_transformation(freq, config)
    transformed_data = transformation.apply(data, is_train=False)

    # We create a test Instance splitter to sample the very last
    # context window from the dataset provided.
    instance_sampler = create_instance_splitter(config, "test")

    # We apply the transformations in test mode
    testing_instances = instance_sampler.apply(transformed_data, is_train=False)
    
    return as_stacked_batches(
        testing_instances,
        batch_size=batch_size,
        output_type=torch.tensor,
        field_names=PREDICTION_INPUT_NAMES,
    )


train_dataloader = create_train_dataloader(
    config=config,
    freq=freq,
    data=train_dataset,
    batch_size=256,
    num_batches_per_epoch=100,
)

test_dataloader = create_backtest_dataloader(
    config=config,
    freq=freq,
    data=test_dataset,
    batch_size=64,
)






# 测试代码 main 函数
def main():
    batch = next(iter(train_dataloader))
    for k, v in batch.items():
        print(k, v.shape, v.type())
        
    outputs = model(
        past_values=batch["past_values"],
        past_time_features=batch["past_time_features"],
        past_observed_mask=batch["past_observed_mask"],
        static_categorical_features=batch["static_categorical_features"]
        if config.num_static_categorical_features > 0
        else None,
        static_real_features=batch["static_real_features"]
        if config.num_static_real_features > 0
        else None,
        future_values=batch["future_values"],
        future_time_features=batch["future_time_features"],
        future_observed_mask=batch["future_observed_mask"],
        output_hidden_states=True,
    )
    print("Loss:", outputs.loss.item())

if __name__ == "__main__":
    main()
