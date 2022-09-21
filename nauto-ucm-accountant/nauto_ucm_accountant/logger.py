import logging


LOG_FORMAT = '%(asctime)s : %(levelname)s : %(name)s:%(lineno)s : %(message)s'


def get_logger() -> logging.Logger:
    logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)
    _logger = logging.getLogger(__name__)
    return _logger


logger = get_logger()
