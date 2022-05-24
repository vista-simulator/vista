""" Logging utilities of the simulator. """
import os
from typing import Union
import inspect
import logging
from functools import partial

DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL


class CustomFormatter(logging.Formatter):
    """ A custom logger class inherited from python logging module. It basically
    define a custom formatting as ``<time>::<level>::<module>::<msg>`` with colored
    text (``DEBUG``: cyan, ``INFO``: regular, ``WARNING``: yellow, ``ERROR``: bold red,
    ``CRITICAL``: underlined bold red). Otherwise, it can be used as if using regular
    python logging.

    Example Usage::

        >>> from vista.utils import logging
        >>> logging.setLevel(logging.ERROR)
        >>> logging.debug('This is a debug message.')
        >>> logging.warning('This is a warning message.')

    """
    PURPLE = '\033[95m'
    RED = "\033[0;31m"
    BOLD_RED = "\033[1;31m"
    BLUE = "\033[0;34m"
    CYAN = "\033[0;36m"
    GREEN = "\033[0;32m"
    YELLOW = '\033[93m'
    BOLD = "\033[;1m"
    UNDERLINE = '\033[4m'
    ENDC = '\033[0;0m'
    format = '%(asctime)s::%(levelname)s::%(message)s'

    FORMATS = {
        logging.DEBUG: CYAN + format + ENDC,
        logging.INFO: format + ENDC,
        logging.WARNING: YELLOW + format + ENDC,
        logging.ERROR: BOLD_RED + format + ENDC,
        logging.CRITICAL: UNDERLINE + BOLD_RED + format + ENDC
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


# Default logging config
logger = logging.getLogger('Vista')
logger.setLevel(logging.WARNING)
ch = logging.StreamHandler()
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)


def setLevel(level: Union[str, int]) -> None:
    """ Set the logging level of this handler. Check CustomFormatter
    for example usage.

    Args:
        level (str or int): Logging level.

    """
    logger.setLevel(level)


def base(msg: str, func: str):
    filename = os.path.splitext(inspect.stack()[1].filename)[0].split('/')
    idx = [i for i, v in enumerate(filename) if v == 'vista']
    if len(idx) > 0:
        idx = idx[-1]
        filename = '.'.join(filename[idx:])
    else:
        filename = filename[0]
    func_name = inspect.stack()[1].function
    caller_name = '.'.join([filename, func_name])
    func = getattr(logger, func)
    func('[{}] {}'.format(caller_name, msg))


error = partial(base, func='error')
warning = partial(base, func='warning')
info = partial(base, func='info')
debug = partial(base, func='debug')
critical = partial(base, func='critical')
