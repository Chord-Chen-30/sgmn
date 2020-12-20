import os
import sys

from osutils import mkdir_if_missing


class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


from datetime import datetime, timedelta
import random
import pickle
import numpy as np
from numpy import mean, std


class UnderwaterLogger():

    def __init__(self, save_path):
        self.save_path = save_path
        self.hyper_params = {}

        time = datetime.now() + timedelta(hours=8)  # Convert UTC to Beijing
        self.time = time.strftime('%Y-%m-%d_%H.%M.%S') + str(round(random.random(), 3))[1:]

        self.save_path += (self.time + '.pkl')

    def hyper_param_info(self, hyper_param, value):
        self.hyper_params[hyper_param] = value

    def save(self):
        pickle.dump(self, open(self.save_path, 'wb'))

