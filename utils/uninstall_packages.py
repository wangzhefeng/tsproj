# -*- coding: utf-8 -*-

# ***************************************************
# * File        : uninstall_packages.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-12-10
# * Version     : 1.0.121016
# * Description : description
# * Link        : https://www.jianshu.com/p/81bffb457ac4
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = []

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def check_dependents(module_name):
    """
    查看一个模块的依赖
    :param module_name: 模块名
    :return: module_name 所依赖的模块的列表
    """
    with os.popen('pip show %s' % module_name) as p:
        dependents = p.readlines()
        if not len(dependents):
            return None
        dependents = dependents[-1]
        dependents = dependents.split(':')[-1].replace(' ', '').strip()
        if dependents:
            dependents = dependents.split(',')
        else:
            dependents = None
        return dependents


def remove(module_name):
    """
    递归地卸载一个模块
    :param module_name: 模块名称
    :return: None
    """
    dependents = check_dependents(module_name)
    if dependents:
        for package in dependents:
            remove(package)
    os.system('pip uninstall -y %s' % module_name)



# 测试代码 main 函数
def main():
    pkg_name = input('请输入要卸载的第三方模块包: ')
    remove(pkg_name)

if __name__ == "__main__":
    main()
