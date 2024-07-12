""" Helper functions that are utilized throughout the main code
"""

from typing import List


def flatten_dicts(dictlist: List[dict]) -> dict:
    """Flatten a one-level nested dict structure

    Example:
        in: [{'n': 3}, {'random_state': 1}]
        out: {'n': 3, 'random_state': 1}

    Args:
        dictlist (dict): A list containing at least one dictionary

    Returns:
        dict: Flattened dict
    """
    return {k: v for d in dictlist for k, v in d.items()}
