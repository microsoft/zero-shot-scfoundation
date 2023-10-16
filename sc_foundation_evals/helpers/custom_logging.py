## Copyright (c) Microsoft Corporation.
## Licensed under the MIT license.
import logging
# colorful logging
from colorlog import ColoredFormatter

LOG_LEVEL = logging.DEBUG

LOGFORMAT = "%(log_color)s%(levelname)-8s%(reset)s | %(asctime)s | %(log_color)s%(message)s%(reset)s"

logging.root.setLevel(LOG_LEVEL)
formatter = ColoredFormatter(LOGFORMAT, datefmt = '%Y-%m-%d %H:%M:%S')
stream = logging.StreamHandler()
stream.setLevel(LOG_LEVEL)
stream.setFormatter(formatter)
log = logging.getLogger('pythonConfig')
log.setLevel(LOG_LEVEL)
log.addHandler(stream)