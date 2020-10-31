# -*- coding: utf-8 -*-

from loguru import logger
import sys

class LogType:
    MAIN = 1
    PRODUCER = 2
    MODEL = 3
    MCTS = 4

logger.add("log/main.log", level="INFO", filter=lambda r: r["extra"].get("type") == LogType.MAIN)
logger.add("log/producer.log", level="INFO", filter=lambda r: r["extra"].get("type") == LogType.PRODUCER)
logger.add("log/model.log", level="INFO", filter=lambda r: r["extra"].get("type") == LogType.MODEL)
logger.add("log/mcts.log", level="INFO", filter=lambda r: r["extra"].get("type") == LogType.MCTS)
main_logger = logger.bind(type=LogType.MAIN)
producer_logger = logger.bind(type=LogType.PRODUCER)
model_logger = logger.bind(type=LogType.MODEL)
mcts_logger = logger.bind(type=LogType.MCTS)

logger.disable("mcts")