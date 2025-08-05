from logging.handlers import TimedRotatingFileHandler
import logging
import os
import time
from pathlib import Path

LOG_DIR = "logs"
LOG_FILE = "app.log"
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("app")  # 글로벌 로거 1개
logger.setLevel("INFO")

if not logger.hasHandlers():
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 콘솔 로그 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 로그 파일 핸들러
    file_handler = TimedRotatingFileHandler(
        filename=os.path.join(LOG_DIR, LOG_FILE),
        when="midnight",      # 매일 자정
        interval=1,           # 1일 간격
        backupCount=7,        # 7일치 로그만 보관
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def get_logger(module_name:str =None) :
     return logger.getChild(module_name) if module_name else logger
    
    