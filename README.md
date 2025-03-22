
<details><summary>目录</summary><p>

- [TODO](#todo)
- [文章](#文章)
- [项目开发步骤](#项目开发步骤)
    - [步骤 1：确定代码框架](#步骤-1确定代码框架)
    - [步骤 2：定义命令行解析](#步骤-2定义命令行解析)
    - [步骤 3：确定调参工具](#步骤-3确定调参工具)
    - [步骤4：减少随机性](#步骤4减少随机性)
- [模型运行记录](#模型运行记录)
    - [项目基本结构](#项目基本结构)
    - [模型运行步骤](#模型运行步骤)
    - [参数配置](#参数配置)
        - [设置随机数](#设置随机数)
        - [设备参数](#设备参数)
        - [任务类型参数](#任务类型参数)
        - [数据参数](#数据参数)
        - [模型定义参数](#模型定义参数)
        - [模型输入、输出参数](#模型输入输出参数)
        - [模型训练参数](#模型训练参数)
        - [模型验证参数](#模型验证参数)
        - [模型测试参数](#模型测试参数)
        - [日志记录参数](#日志记录参数)
    - [模型运行](#模型运行)
        - [参数配置](#参数配置-1)
        - [创建任务实例](#创建任务实例)
        - [模型训练、验证](#模型训练验证)
        - [模型测试](#模型测试)
</p></details><p></p>

# TODO

* [ ] 时间序列预测预测、目标特征标准化；
* [ ] 深度学习模型推理；
* [ ] 时间序列预测机器学习模型融合；

# 文章

* [一文梳理Transformer在时间序列预测中的发展历程代表工作](https://mp.weixin.qq.com/s/OjK7Q7DSoTM_p1MLye9RWw)
* [圆圆的算法笔记](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzIyOTUyMDIwNg==&action=getalbum&album_id=2339781350876332033&scene=173&from_msgid=2247487281&from_itemidx=1&count=3&nolastread=1#wechat_redirect)
* [时序人1](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=Mzg3NDUwNTM3MA==&action=getalbum&album_id=1565545072782278657&scene=173&from_msgid=2247484974&from_itemidx=1&count=3&nolastread=1#wechat_redirect)
* [时序人2](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=Mzg3NDUwNTM3MA==&action=getalbum&album_id=1588681516295979011&scene=173&from_msgid=2247484974&from_itemidx=1&count=3&nolastread=1#wechat_redirect)
* [时序人3](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzU4NTA1MDk4MA==&action=getalbum&album_id=1401576921242337281&scene=173&from_msgid=2247526075&from_itemidx=2&count=3&nolastread=1#wechat_redirect)
* [时间序列预报](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzkzMTMyMDQ0Mw==&action=getalbum&album_id=2512078794435133440&scene=173&from_msgid=2247485132&from_itemidx=2&count=3&nolastread=1#wechat_redirect)
* Informer
    - [知乎](https://zhuanlan.zhihu.com/p/355133560)
    - [知乎](https://zhuanlan.zhihu.com/p/499399526)
    - [GitHub](https://github.com/zhouhaoyi/Informer2020)
    - [Paper](https://arxiv.org/abs/2012.07436)
* [CL-Timeseries](https://github.com/kashif/CL_Timeseries)
* [LSTNet](https://github.com/laiguokun/LSTNet)
* [TSForecasting](https://github.com/rakshitha123/TSForecasting)
* TFT
    - [知乎](https://zhuanlan.zhihu.com/p/514287527)
    - [GitHub](https://github.com/google-research/google-research/tree/master/tft)
    - [Complete Tutorial](https://towardsdatascience.com/temporal-fusion-transformer-time-series-forecasting-with-deep-learning-complete-tutorial-d32c1e51cd91)
    - [公众号](https://mp.weixin.qq.com/s/0AXSOgivCytHKTCmpPJfFg)
* [多任务学习MTL模型：MMoE、PLE](https://zhuanlan.zhihu.com/p/425209494)
* [Temporal Pattern Attention for Multivariate Time Series Forecasting](https://github.com/shunyaoshih/TPA-LSTM)
* [pytorch-forecasting](https://github.com/jdb78/pytorch-forecasting)
* [pytorch-ts](https://github.com/zalandoresearch/pytorch-ts)
* [HF-Time Series Transformer](https://huggingface.co/docs/transformers/main/en/model_doc/time_series_transformer)

# 项目开发步骤

## 步骤 1：确定代码框架

首先确定好具体任务，然后根据任务选择合适的框架，如 `PyTorch Lightning` 或 `MMDection`。
如果框架有默认目录，则遵守。否则可以创建适合自己的目录，一般而言目录推荐如下：

* `general`：常见的训练过程、保存加载模型过程，与具体任务相关的代码
* `layers`：模型定义、损失函数等
* `models`: 模型定义、损失函数等
* `experiments`：具体任务的训练流程、数据读取和验证过程

```
general/
│   train.py
│   task.py
│   mutils.py
layers/
experiments/
│   task1/
│        train.py
│        task.py
│        eval.py
│        dataset.py
│   task2/
│        train.py
│        task.py
│        eval.py
│        dataset.py
```

## 步骤 2：定义命令行解析

Notebook 虽然很好用，但是具体 `.py` 代码实际运行和管理更加方便。所以命令行解析就非常关键。
可以选择自己喜欢的参数解析器，在命令行中一般推荐加入**学习率**、**batch size**、**seed** 等超参数。

```bash
$ python train.py --learning ... --seed ... --hidden_size ...
```

## 步骤 3：确定调参工具

在调试和训练模型的过程中，肯定需要多次训练，此时 TensorBoard 可以非常好的管理实验日志。

调参是非常乏味的，比较重要的是确定好**学习率**和 **batch size**。
**学习率**和**优化器**有非常多的选择，SGD 是一个比较好的开始。
一般而言模型越深，学习率越小；batch size 越大，学习率越大。

## 步骤4：减少随机性

深度学习模型有一定的随机性，模型是否可复现非常重要。在比赛期间，
非常推荐提前把不同 fold 的次序存储到文件，减少随机性。

把配置文件、模型权重、日志文件保存好，这样每次都可以进行实验对比。

* PyTorch 设置 SEED

```python
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
```

* TF 1.X 设置 SEED

```python
from tfdeterminism import patch
patch()
os.environ['PYTHONHASHSEED']=str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.set_random_seed(SEED)
```

* TF 2.X 设置 SEED

```python
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED']=str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
```

# 模型运行记录

## 项目基本结构

* dataset
* data_provider
* layers
* models
* exp
* scripts
* `run.py`
* `run.sh`

## 模型运行步骤

1. 参数配置
    - 设置随机数
    - 设备配置
    - 任务类型参数
    - 数据参数
    - 模型输入、输出参数
    - 模型定义参数
    - 模型训练参数
    - 模型验证参数
    - 模型测试参数
    - 日志记录参数
2. 模型运行
    - `run.py`


## 参数配置

### 设置随机数

```python
# 设置随机数
fix_seed = 2021
random.seed(fix_seed)
np.random.seed(fix_seed)
torch.manual_seed(fix_seed)
torch.cuda.manual_seed_all(fix_seed)
torch.backends.cudnn.deterministic = True
```

### 设备参数

```python
# 设备配置
use_gpu = True if torch.cuda.is_available() else False
use_multi_gpu = False
devices = "0,1,2,3"
device_ids = [int(id_) for id_ in devices.replace(' ', '').split(',')]
if use_gpu and use_multi_gpu:
    gpu = devices
elif use_gpu and not use_multi_gpu:
    gpu = device_ids[0]
print(f"use_gpu: {use_gpu}")
print(f"use_multi_gpu: {use_multi_gpu}")
print(f"gpu: {gpu}")
```

```python
# class Config
# 设备参数
use_gpu = use_gpu
use_multi_gpu = use_multi_gpu
devices = devices
device_ids = device_ids
gpu = gpu
args.num_workers = 10
```

### 任务类型参数

```python
# class Config
# 任务类型参数
task_name = "long_term_forecast"
is_training = True
```

### 数据参数

```python
# class Config
# TODO 数据参数
root_path = "dataset/ETT-small/"
data_path = "ETTm1.csv"
data = "ETTm1"
freq = "m"
```

### 模型定义参数

```python
# class Config
# 模型定义参数
model_id = "ETTm1_96_96"
model = "TimesNet"
features = "M"
target = "OT"
e_layers = 2
d_layers = 1
factor = 3
enc_in = 7
dec_in = 7
num_kernels = 6
c_out = 7
d_ff = 64
des = "Exp"
embed = "fixed"
```

### 模型输入、输出参数

```python
# 模型输入、输出参数
seq_len = 96
label_len = 48
pred_len = 96
d_model = 64
dropout = 0.1
top_k = 5
```

### 模型训练参数

```python
iters = 1
train_epochs = 10
batch_size = 32
learning_rate = 0.0001
patience = 3
checkpoints = "checkpoints/"
```

### 模型验证参数

```python

```

### 模型测试参数

```python

```

### 日志记录参数

```python

```

## 模型运行

```python
# python libraries
import random
import numpy as np
import torch

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_imputation import Exp_Imputation
from exp.exp_classification import Exp_Classification
```

### 参数配置

```python
# 设置随机数
fix_seed = 2021
random.seed(fix_seed)
np.random.seed(fix_seed)
torch.manual_seed(fix_seed)
torch.cuda.manual_seed_all(fix_seed)
torch.backends.cudnn.deterministic = True

# 设备配置
use_gpu = True if torch.cuda.is_available() else False
use_multi_gpu = False
devices = "0,1,2,3"
device_ids = [int(id_) for id_ in devices.replace(' ', '').split(',')]
if use_gpu and use_multi_gpu:
    gpu = devices
elif use_gpu and not use_multi_gpu:
    gpu = device_ids[0]
print(f"use_gpu: {use_gpu}")
print(f"use_multi_gpu: {use_multi_gpu}")
print(f"gpu: {gpu}") 
print("-" * 80)


class Config:
    pass
```

### 创建任务实例

```python
args = Config()
for ii in range(args.iters):
    args.ii = ii
    exp = Exp_Long_Term_Forecast(args)
```

1. `exp = Exp_Long_Term_Forecast(args)`
    - `args.model`
2. `Exp_Basic`
    - `__init__`
        - `exp.args`
        - `exp.model_dict`
    - `_acquire_device`
        - `exp.device`
    - `_build_model`
3. `Exp_Long_Term_Forecast`
    - `_build_model`
        - `exp.model`：多 GPU 训练
        - `models.XXX.Model()`
            - `exp.model.task_name`
            - `exp.model.seq_len`
            - `exp.model.pred_len`
            - `exp.model.`others params 

### 模型训练、验证

```python
args = Config()
for ii in range(args.iters):
    args.ii = ii
    setting = f""
    exp.train(setting)
    exp.val(setting)
    # 清除 CUDA 缓存
    torch.cuda.empty_cache()
```

4. `exp.train(setting)`
    - `exp._get_data()`
        - `data_provider.data_factor.data_provider`
    - 记录训练开始时间
        - `time_now`
    - 计算训练数据长度
        - `train_steps`
    - 早停类实例
        - `early_stopping`
            - args.patience
    - 模型优化器
        - `model_optim`
            - `exp.model.parameters()`
            - `args.learning_rate`
    - 模型损失函数
        - `criterion`
    - 自动混合精度运算实例
        - `scaler`
    - 模型训练
        - args.train_epochs：分 epoch 训练
            - `train_loss` 记录
            - `exp.model.train() -> models.XXX.Model().fowrad() -> models.XXX.Model().forecast()`
            - `batch_x, batch_y, batch_x_mark, batch_y_mark`：分 batch 训练
                - `model_optim.zero_grad()`
                - 数据预处理
                - 前向传播
                - 后向传播
                - 日志打印
                    - 计算平均训练误差
                    - 计算验证数据误差、测试数据误差
            - 日志打印
            - 早停机制
            - 学习率调整
    - 最优模型保存

### 模型测试

```python
args = Config()
args.ii = 0
exp = Exp_Long_Term_Forecast(args)
setting = f""
exp.test(setting, test = 1)
# 清除 CUDA 缓存
torch.cuda.empty_cache()
```
