# logging_config.py
import logging
from logging import StreamHandler, FileHandler
from logging import Formatter

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

def setup_logging(
    level: int = logging.INFO,
    console: bool = True,
    logfile: str | None = "app.log",
) -> None:
    logger = logging.getLogger()        # root
    if logger.handlers:
        return

    logger.setLevel(level)

    formatter = Formatter(LOG_FORMAT)

    if console:
        ch = StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if logfile is not None:
        fh = FileHandler(logfile, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
