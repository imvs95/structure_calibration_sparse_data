import logging
from time import time
from typing import Any


def sharedmemory(f):
    def wrap(*args, **kwargs):
        ns = kwargs.pop("ns")
        ret = f(*args, **kwargs)
        ns.result = ret
        return ret

    return wrap


def timing(f):
    """Wrapper decorator

    Args:
        f (function): A function to be wrapped
    """

    def wrap(*args, **kwargs) -> Any:
        """Wrapper that times the duration of a wrapperd function

        Returns:
            res (Any): The output of function f
        """

        time1 = time()
        ret = f(*args, **kwargs)
        time2 = time()
        min, sec = divmod(time2 - time1, 60)

        # Ugly class name hack :-)
        function_name = str(f).split(" ")[1]
        log_text = (
            "\033[1m{:s}\033[0m function took {:.0f} minutes and {:.3f} seconds".format(
                function_name, min, sec
            )
        )
        logging.info(log_text)

        # Update logger
        return ret

    return wrap
