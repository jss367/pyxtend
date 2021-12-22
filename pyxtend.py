from typing import Iterable


def struct(obj, limit=3):
    "Return the general structure of any object"
    if isinstance(obj, str) or not isinstance(obj, Iterable):
        return type(obj)

    #     a = [struct(obj_) for obj_ in obj[:limit]]
    c = iter(obj)
    q = []

    for i in range(min(limit, len(obj))):
        q.append(type(next(c, None)))
    a = [struct(obj_) for obj_ in obj]
    a = q
    if len(obj) > limit:
        a.append(f"...{len(obj)} total")
    return {type(obj): a}
