from collections.abc import Iterable
from typing import Any, Union
import pandas as pd

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
    elif isinstance(obj, str):
        return "str"
    elif obj_type_name in ["Tensor", "EagerTensor"]:
        return {obj_type_name: [f"{obj.dtype}, shape={tuple(getattr(obj, 'shape', ()))}"]}
    elif obj_type_name == "ndarray":
        inner_structure = "empty" if obj.size == 0 else struct(obj.item(0), level + 1)
        shape = tuple(obj.shape)
        dtype = obj.dtype.name
        return {f"{type(obj).__name__}": [f"{dtype}, shape={shape}"]}
    elif obj_type_name == "Polygon":
        coords = list(getattr(obj, "exterior", {}).coords) if hasattr(obj, "exterior") else []
        shape = (len(coords), len(coords[0]) if coords else 0)
        return {f"{type(obj).__name__}": [f"float64, shape={shape}"]}
    elif isinstance(obj, torch.nn.Module):
        # Get all parameters of the nn.Module
        params = list(obj.named_parameters())
        if examples:
            inner_structure = {name: struct(param.data, level + 1, limit, examples) for name, param in params}
        else:
            inner_structure = {name: struct(param.data, level + 1, limit, examples) for name, param in params}
        return {type(obj).__name__: inner_structure}
    elif isinstance(obj, pd.core.groupby.generic.DataFrameGroupBy):
        groupby_summary(obj)
    elif isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
        if level < limit:
            if examples:
                inner_structure = [x for x in obj]
            else:
                inner_structure = [struct(x, level + 1) for x in obj]
            if len(obj) > 3:
                inner_structure = inner_structure[:3] + [f"...{len(obj)} total"]
            return {type(obj).__name__: inner_structure}
        else:
            return {type(obj).__name__: "..."}
    else:
        return "unsupported"




def groupby_summary(groupby_object):
    """
    Provide a comprehensive summary of a DataFrameGroupBy object, focusing on key statistics,
    examples of groups, and overall group description including total items and average items per group.

    Parameters:
    - groupby_object: A pandas DataFrameGroupBy object.

    Returns:
    None
    """
    # Total number of groups and total items across all groups
    num_groups = groupby_object.ngroups
    total_items = sum(len(group) for _, group in groupby_object)
    average_items_per_group = total_items / num_groups if num_groups > 0 else 0

    print(f"Total number of items: {total_items}")
    print(f"Total number of groups: {num_groups}")
    print(f"Average number of items per group: {average_items_per_group:.2f}\n")

    # Summary statistics of group sizes
    group_sizes = groupby_object.size()
    print("Group sizes summary:")
    print(group_sizes.describe())
    print("\n")

    print("Examples of groups (first row per group):")
    # Initialize a counter
    examples_shown = 0
    # Iterate over the groupby object and print the first row of the first n_examples groups
    for _, group in groupby_object:
        print(group.head(1))
        examples_shown += 1
        if examples_shown >= examples_shown:
            break
    print("\n")


    # Global aggregated statistics for numeric columns (mean, median)
    print("Global mean values for numeric columns:")
    try:
        # Explicitly specify numeric_only=True
        print(groupby_object.mean(numeric_only=True).mean(numeric_only=True))  # Mean of means for each group
    except TypeError:
        print("No numeric columns to calculate mean.")
    print("\n")

    print("Global median values for numeric columns:")
    try:
        # Explicitly specify numeric_only=True
        print(groupby_object.median(numeric_only=True).median(numeric_only=True))  # Median of medians for each group
    except TypeError:
        print("No numeric columns to calculate median.")
