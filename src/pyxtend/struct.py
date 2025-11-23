import itertools
from collections.abc import Iterable, Sized
from typing import Any, Union


def _create_ndarray_summary(obj) -> str:
    """Create a summary string for a numpy ndarray."""
    shape = tuple(obj.shape)
    dtype = obj.dtype.name
    return f"{dtype}, shape={shape}"


def _create_ndarray_structure(obj, examples: bool, preview_limit: int) -> dict:
    """Create structure dictionary for a numpy ndarray."""
    summary = _create_ndarray_summary(obj)
    if not examples:
        return {f"{type(obj).__name__}": [summary]}

    flat = obj.flatten()
    preview = flat[:preview_limit].tolist()

    if flat.size > preview_limit:
        preview.append(f"...{flat.size} total")

    return {f"{type(obj).__name__}": [summary, *preview]}


def _create_polygon_structure(obj) -> dict:
    """Create structure dictionary for a Polygon object."""
    coords = list(getattr(obj, "exterior", {}).coords) if hasattr(obj, "exterior") else []
    shape = (len(coords), len(coords[0]) if coords else 0)
    return {f"{type(obj).__name__}": [f"float64, shape={shape}"]}


def _create_tensor_structure(obj, obj_type_name: str) -> dict:
    """Create structure dictionary for Tensor/EagerTensor objects."""
    return {obj_type_name: [f"{obj.dtype}, shape={tuple(getattr(obj, 'shape', ()))}"]}


def _create_iterable_structure(obj, level: int, limit: int, examples: bool) -> dict:
    """Create structure dictionary for iterable objects."""
    if level >= limit:
        return {type(obj).__name__: "..."}

    preview_limit = limit
    iterator = iter(obj)
    items = list(itertools.islice(iterator, preview_limit + 1))
    truncated = len(items) > preview_limit
    items = items[:preview_limit]

    inner_structure = items if examples else [struct(x, level + 1, limit, examples) for x in items]

    summary_entry = None
    if isinstance(obj, Sized):
        total = len(obj)
        if total > preview_limit:
            summary_entry = f"...{total} total"
    elif truncated:
        summary_entry = "...more"

    if summary_entry:
        inner_structure.append(summary_entry)

    return {type(obj).__name__: inner_structure}


def _create_custom_object_structure(obj, obj_type_name: str, level: int, limit: int, examples: bool) -> dict:
    """Create structure dictionary for custom objects."""
    attributes = {
        key: struct(getattr(obj, key), level + 1, limit, examples) for key in dir(obj) if not key.startswith("_")
    }
    return {obj_type_name: attributes}


def struct(obj: Any, level: int = 0, limit: int = 3, examples: bool = False) -> Union[str, dict]:
    """
    Returns the general structure of a given Python object.

    Args:
        obj: The Python object to analyze.
        level: The current depth of recursion (default: 0).
        limit: The maximum number of elements to display for each type (default: 3).
        examples: Whether to include examples of elements in the returned structure (default: False).

    Returns:
        The structure of the input object as a dictionary or string.
    """
    obj_type_name = type(obj).__name__

    if isinstance(obj, (int, float, bool)):
        return obj_type_name
    if isinstance(obj, str):
        return "str"
    if obj_type_name in ["Tensor", "EagerTensor"]:
        return _create_tensor_structure(obj, obj_type_name)
    if obj_type_name == "ndarray":
        return _create_ndarray_structure(obj, examples, limit)
    if obj_type_name == "Polygon":
        return _create_polygon_structure(obj)
    if obj_type_name == "DataFrameGroupBy":
        return groupby_summary(obj)
    if isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
        return _create_iterable_structure(obj, level, limit, examples)
    return _create_custom_object_structure(obj, obj_type_name, level, limit, examples)


def groupby_summary(groupby_object):
    """
    Provide a comprehensive summary of a DataFrameGroupBy object, focusing on key statistics,
    examples of groups, and overall group description including total items and average items per group.

    Parameters:
    - groupby_object: A pandas DataFrameGroupBy object.

    Returns:
        - summary_data: A dictionary with many keys:
    """
    # Total number of groups and total items across all groups
    num_groups = groupby_object.ngroups
    total_items = sum(len(group) for _, group in groupby_object)
    average_items_per_group = total_items / num_groups if num_groups > 0 else 0

    # Summary statistics of group sizes
    group_sizes = groupby_object.size()

    group_examples = {name: group.head(2) for name, group in groupby_object}

    # Global aggregated statistics for numeric columns (mean, median)
    try:
        # Explicitly specify numeric_only=True
        global_mean_values = groupby_object.mean(numeric_only=True).mean(
            numeric_only=True
        )  # Mean of means for each group
    except TypeError:
        global_mean_values = "No numeric columns to calculate mean."

    try:
        # Explicitly specify numeric_only=True
        global_median_values = groupby_object.median(numeric_only=True).median(
            numeric_only=True
        )  # Median of medians for each group
    except TypeError:
        global_median_values = "No numeric columns to calculate median."

    return {
        "total_items": total_items,
        "num_groups": num_groups,
        "average_items_per_group": average_items_per_group,
        "group_size_stats": group_sizes.describe().to_dict(),  # Convert Series to dictionary
        "group_examples": group_examples,
        "mean_values": global_mean_values,
        "median_values": global_median_values,
    }
