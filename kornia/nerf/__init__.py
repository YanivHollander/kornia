import logging
import sys


class NerfLogger(logging.Logger):
    def exception(self, msg, *args, exc_info=True, **kwargs) -> None:
        super().exception(msg, *args, exc_info=exc_info, **kwargs)
        raise msg


def init_logger():
    logging.setLoggerClass(NerfLogger)

    logger = logging.getLogger('NeRF logger')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


logger = init_logger()
