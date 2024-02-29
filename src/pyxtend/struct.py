from collections.abc import Iterable
from typing import Any, Union


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




def concise_groupby_summary(groupby_object):
    """
    Provide a concise summary of a DataFrameGroupBy object, focusing on key statistics
    and a couple of example groups.

    Parameters:
    - groupby_object: A pandas DataFrameGroupBy object.

    Returns:
    None
    """
    # Total number of groups
    print(f"Total groups: {len(groupby_object)}\n")

    # Summary statistics of group sizes
    group_sizes = groupby_object.size()
    print("Group sizes summary:")
    print(group_sizes.describe())
    print("\n")

    # Display a couple of example groups (first two groups as examples)
    example_groups = groupby_object.apply(lambda x: x.head(1))  # Adjust based on the size and readability of your data
    print("Examples of groups (first row per group):")
    print(example_groups.head(2))
    print("\n")

    # Global aggregated statistics for numeric columns (mean, median)
    print("Global mean values for numeric columns:")
    try:
        print(groupby_object.mean().mean())  # Mean of means for each group
    except TypeError:
        print("No numeric columns to calculate mean.")
    print("\n")

    print("Global median values for numeric columns:")
    try:
        print(groupby_object.median().median())  # Median of medians for each group
    except TypeError:
        print("No numeric columns to calculate median.")
