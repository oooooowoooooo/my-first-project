import logging
import os
from datetime import datetime


def setup_logger():
    """配置日志系统（跨平台兼容）"""
    # 创建日志目录
    log_dir = "logs"
    try:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    except Exception as e:
        print(f"创建日志目录失败：{e}")

    # 日志文件名（按日期分割）
    log_file = os.path.join(log_dir, f"dailytaskai_{datetime.now().strftime('%Y%m%d')}.log")

    # 配置日志格式
    log_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(log_format)
    file_handler.setLevel(logging.INFO)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    console_handler.setLevel(logging.INFO)

    # 配置根日志器
    logger = logging.getLogger("DailyTaskAI")
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # 避免重复添加处理器
    logger.propagate = False

    return logger


# 初始化日志器
logger = setup_logger()