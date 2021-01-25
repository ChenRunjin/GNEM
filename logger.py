import logging


def set_logger(path=None):
    logger = logging.getLogger()
    logger.setLevel('DEBUG')
    BASIC_FORMAT = "%(asctime)s:%(levelname)s:%(message)s"
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
    chlr = logging.StreamHandler()
    chlr.setFormatter(formatter)
    chlr.setLevel('INFO')
    logger.addHandler(chlr)
    if path:
        fhlr = logging.FileHandler(path)
        fhlr.setFormatter(formatter)
        logger.addHandler(fhlr)
    return logger

