import contextlib
from collections.abc import Iterable
from typing import Any, Iterable, Union

import numpy as np
import tensorflow as tf
import torch


def struct(obj: Any, level: int = 0, limit: int = 3) -> Union[str, dict]:
    """
    Returns the general structure of a given Python object.

    Args:
        obj: The Python object to analyze.
        level: The current depth of recursion (default: 0).
        limit: The maximum number of elements to display for each type (default: 3).

    Returns:
        The structure of the input object as a dictionary or string.
    """
    if isinstance(obj, (int, float, bool)):
        return type(obj).__name__
    elif isinstance(obj, str):
        return "str"
    elif isinstance(obj, torch.Tensor):
        return {f"{type(obj).__name__}": [f"{obj.dtype}, shape={tuple(obj.shape)}"]}
    elif isinstance(obj, tf.Tensor):
        return {f"{type(obj).__name__}": [f"{obj.dtype.name}, shape={tuple(obj.shape)}"]}
    elif isinstance(obj, np.ndarray):
        if obj.size == 0:  # Check if the array is empty
            inner_structure = "empty"
        else:
            inner_structure = struct(obj.item(0), level + 1)
        return {f"{type(obj).__name__}": [f"{obj.dtype}, shape={obj.shape}"]}
    elif isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
        if level < limit:
            inner_structure = [struct(x, level + 1) for x in obj]
            if len(obj) > 3:
                inner_structure = inner_structure[:3] + [f"...{len(obj)} total"]
            return {type(obj).__name__: inner_structure}
        else:
            return {type(obj).__name__: "..."}
    else:
        return "unsupported"  # type(obj).__name__ or {type(obj).__name__: 'unsupported'}


def vprint(*text: str) -> None:
    """
    Print the text if verbose=True in outer context.
    Print nothing if verbose=False or is undefined.
    """
    with contextlib.suppress(NameError):
        if verbose:
            print(*text)
