import contextlib
from typing import Iterable


def struct(obj, limit=3):
    "Return the general structure of any object"
    if isinstance(obj, str) or not isinstance(obj, Iterable):
        return type(obj)
    if isinstance(obj, set):
        c = iter(obj)
        a = []
        for _ in range(min(limit, len(obj))):
            a.append(type(next(c, None)))
        if len(obj) > limit:
            a.append(f"...{len(obj)} total")
        return {type(obj): a}

    a = [struct(obj_) for obj_ in obj[:limit]]
    if len(obj) > limit:
        a.append(f"...{len(obj)} total")
    return {type(obj): a}


def vprint(*text: str) -> None:
    """
    Print the text if verbose=True in outer context.
    Print nothing if verbose=False or is undefined.
    """
    with contextlib.suppress(NameError):
        if verbose:
            print(*text)
