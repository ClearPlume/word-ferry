import logging

from src.word_ferry.path import get_logs_dir


def setup_logger(name: str, type_: str, level: int = logging.INFO) -> logging.Logger:
    log_dir = get_logs_dir()
    logger = logging.getLogger(f"tabular_sense.{name}.{type_}")
    logger.setLevel(level)

    file_handler = logging.FileHandler(log_dir / name / f"{type_}.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
