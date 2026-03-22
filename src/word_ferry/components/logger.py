import logging

from word_ferry.path import get_logs_dir


def setup_logger(name: str, type_: str, level: int = logging.INFO) -> logging.Logger:
    log_dir = get_logs_dir() / name

    if not log_dir.exists():
        log_dir.mkdir(parents=True)

    logger = logging.getLogger(f"word_ferry.{name}.{type_}")
    logger.setLevel(level)

    file_handler = logging.FileHandler(log_dir / f"{type_}.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
