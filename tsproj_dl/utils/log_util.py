"""日志工具模块

该模块提供了日志记录功能，包括：
- 控制台日志输出
- 按天轮转的文件日志
- 日志级别通过环境变量SERVICE_LOG_LEVEL配置
"""

import os
import sys
import re
import logging
from logging import handlers

# 项目根路径
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 日志级别，默认为INFO
LOG_LEVEL = os.environ.get("SERVICE_LOG_LEVEL", "INFO")

# 默认日志格式
default_formatter = logging.Formatter(
    "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d:%(funcName)s] %(message)s",
)

# 控制台日志处理器
stream_handler = logging.StreamHandler(stream=sys.stderr)
stream_handler.setLevel(LOG_LEVEL)
stream_handler.setFormatter(default_formatter)

# 日志文件路径
log_dir = f"{ROOT_PATH}/logs"
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
log_path = os.path.join(log_dir, "service")


# 按天轮转的文件日志处理器
time_rotating_file_handler = handlers.TimedRotatingFileHandler(
    filename=log_path, when="MIDNIGHT", interval=1, backupCount=10
)
time_rotating_file_handler.suffix = "%Y-%m-%d.log"
# 注意：suffix和extMatch必须匹配，否则过期日志不会被删除
time_rotating_file_handler.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}.log$")
time_rotating_file_handler.setLevel(LOG_LEVEL)
time_rotating_file_handler.setFormatter(default_formatter)

# 主日志记录器
logger = logging.getLogger(__name__)
logger.addHandler(stream_handler)
logger.addHandler(time_rotating_file_handler)
logger.setLevel(LOG_LEVEL)
logger.propagate = False




def main():
    """日志功能演示"""
    # 测试不同级别的日志
    logger.debug("这是一条调试信息")
    logger.info("这是一条普通信息")
    logger.warning("这是一条警告信息")
    logger.error("这是一条错误信息")
    logger.critical("这是一条严重错误信息")
    
    # 测试日志文件轮转
    for i in range(100):
        logger.info(f"测试日志轮转 - {i}")

if __name__ == "__main__":
    main()