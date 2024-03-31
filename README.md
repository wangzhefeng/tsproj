
<details><summary>目录</summary><p>

- [文章](#文章)
- [步骤1：确定代码框架](#步骤1确定代码框架)
- [步骤2：定义命令行解析](#步骤2定义命令行解析)
- [步骤3：确定调参工具](#步骤3确定调参工具)
- [步骤4：减少随机性](#步骤4减少随机性)
</p></details><p></p>

# 文章

* [一文梳理Transformer在时间序列预测中的发展历程代表工作](https://mp.weixin.qq.com/s/OjK7Q7DSoTM_p1MLye9RWw)
* [圆圆的算法笔记](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzIyOTUyMDIwNg==&action=getalbum&album_id=2339781350876332033&scene=173&from_msgid=2247487281&from_itemidx=1&count=3&nolastread=1#wechat_redirect)
* [时序人1](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=Mzg3NDUwNTM3MA==&action=getalbum&album_id=1565545072782278657&scene=173&from_msgid=2247484974&from_itemidx=1&count=3&nolastread=1#wechat_redirect)
* [时序人2](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=Mzg3NDUwNTM3MA==&action=getalbum&album_id=1588681516295979011&scene=173&from_msgid=2247484974&from_itemidx=1&count=3&nolastread=1#wechat_redirect)
* [时间序列预报](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzkzMTMyMDQ0Mw==&action=getalbum&album_id=2512078794435133440&scene=173&from_msgid=2247485132&from_itemidx=2&count=3&nolastread=1#wechat_redirect)
* Informer
    - [知乎](https://zhuanlan.zhihu.com/p/355133560)
    - [知乎](https://zhuanlan.zhihu.com/p/499399526)
    - [GitHub](https://github.com/zhouhaoyi/Informer2020)
    - [Paper](https://arxiv.org/abs/2012.07436)
* [CL-Timeseries](https://github.com/kashif/CL_Timeseries)
* [LSTNet](https://github.com/laiguokun/LSTNet)
* [TSForecasting](https://github.com/rakshitha123/TSForecasting)
* [Time-Series-Library](https://github.com/thuml/Time-Series-Library)
* [Autoformer](https://github.com/thuml/Autoformer)
* TFT
    - [知乎](https://zhuanlan.zhihu.com/p/514287527)
    - [GitHub](https://github.com/google-research/google-research/tree/master/tft)
    - [Complete Tutorial](https://towardsdatascience.com/temporal-fusion-transformer-time-series-forecasting-with-deep-learning-complete-tutorial-d32c1e51cd91)
    - [公众号](https://mp.weixin.qq.com/s/0AXSOgivCytHKTCmpPJfFg)
* [多任务学习MTL模型：MMoE、PLE](https://zhuanlan.zhihu.com/p/425209494)
* TimeNet
    - [知乎](https://zhuanlan.zhihu.com/p/606575441)
* [Temporal Pattern Attention for Multivariate Time Series Forecasting](https://github.com/shunyaoshih/TPA-LSTM)
* [pytorch-forecasting](https://github.com/jdb78/pytorch-forecasting)
* [pytorch-ts](https://github.com/zalandoresearch/pytorch-ts)
* [HF-Time Series Transformer](https://huggingface.co/docs/transformers/main/en/model_doc/time_series_transformer)
* [时序人3](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzU4NTA1MDk4MA==&action=getalbum&album_id=1401576921242337281&scene=173&from_msgid=2247526075&from_itemidx=2&count=3&nolastread=1#wechat_redirect)

# 步骤1：确定代码框架

首先确定好具体任务，然后根据任务选择合适的框架，如 `PyTorch Lightning` 或 `MMDection`

如果框架有默认目录，则遵守。否则可以创建适合自己的目录，一般而言目录推荐如下：

* `general`：常见的训练过程、保存加载模型过程，与具体任务相关的代码
* `layers`：模型定义、损失函数等
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

# 步骤2：定义命令行解析

Notebook 虽然很好用，但是具体 `.py` 代码实际运行和管理更加方便。所以命令行解析就非常关键

你可以选择自己喜欢的参数解析器，在命令行中一般推荐加入学习率、batch、seed 等超参数

```bash
$ python train.py --learning ... --seed ... --hidden_size ...
```

# 步骤3：确定调参工具

在调试和训练模型的过程中，肯定需要多次训练，此时 TensorBoard 可以非常好的管理实验日志。

调参是非常乏味的，比较重要的是确定好学习率和 batch size。学习率和优化器有非常多的选择，
SGD 是一个比较好的开始。一般而言模型越深，学习率越小。batch size 越大，学习率越大

# 步骤4：减少随机性

深度学习模型有一定的随机性，模型是否可复现非常重要。在比赛期间，非常推荐提前把不同 fold 的次序存储到文件，
减少随机性

把配置文件、模型权重、日志文件保存好，这样每次都可以进行实验对比

* PyTorch 设置 SEED

```python
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
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
