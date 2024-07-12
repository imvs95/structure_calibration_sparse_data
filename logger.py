import sys
import logging


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, message):
        self.log.write(message)

    def flush(self):
        self.log.flush()
