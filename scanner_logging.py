import logging
import sys

def setup_logging():
    logger = logging.getLogger()
    logger_formatter = logging.Formatter('%(asctime)s | %(filename)20s:%(lineno)-4s | %(levelname).4s | %(message)s', '%m-%d %H:%M:%S')
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logger_formatter)
    logger.addHandler(console_handler)
