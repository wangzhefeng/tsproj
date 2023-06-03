# -*- coding: utf-8 -*-

# ***************************************************
# * File        : TSPredDataset.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-05-21
# * Version     : 0.1.052117
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
import warnings

import paddle

warnings.filterwarnings("ignore")

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# !paddlepaddle
class TSPredDataset(paddle.io.Dataset):
    """
    时序预测 Dataset 

    划分数据集、适配dataloader所需的dataset格式
    ref: https://github.com/thuml/Autoformer/blob/main/data_provider/data_loader.py
    """

    def __init__(self) -> None:
        super(TSPredDataset, self).__init__()





# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
