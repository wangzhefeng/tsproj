# -*- coding: utf-8 -*-

# ***************************************************
# * File        : enum_tools.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-06-01
# * Version     : 0.1.060122
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
from enum import Enum

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class TypeFlag(Enum):
    """
    数据用途编码
    """
    train = 0  # 训练数据
    val = 1  # 验证数据
    test = 2  # 测试数据  


def GetEnumInfo(enum_cls, info):
    """
    获取枚举类型的
    """
    if enum_cls._value2member_map_.get(info) is not None:
        return enum_cls._value2member_map_[info]
    elif enum_cls._member_map_.get(info) is not None:
        return enum_cls._member_map_[info]
    else:
        return None




# 测试代码 main 函数
def main():
    type_name = GetEnumInfo(TypeFlag, "train")
    print(type_name.name)
    print(type_name.value)

if __name__ == "__main__":
    main()
