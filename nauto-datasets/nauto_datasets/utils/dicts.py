from typing import Any, Dict, List, Optional, Tuple, TypeVar, DefaultDict, Callable

from collections import defaultdict


K = TypeVar('K')
V = TypeVar('V')


def flatten_nested_dict(
        nested_dict: Dict[str, V],
        key_prefix: Optional[str] = None,
        sep: str = '/'
) -> Dict[str, V]:
    """Flattens potentially nested dictionary by concatenating string keys
    on the path to values at the leaves.

    Given a nested dictionary e.g.:
    ```
    {
       a: {
          b: {
             c: value,
             ...
          },
          ...
       },
       ...
    }
    ```
    returns a flat dictionary with keys in the form:
    ```
    {
       a/b/c: value,
       ...
    }
    ```

    Args:
        nested_dict: a potentially nested dictionary of features
        key_prefix: when provided, each of the keys in dictionary will
            be prefixed with 'key_prefix/'
        sep: a separator used to glue together names on the path to a leaf

    Returns:
       a flat dictionary of the form `key-path` -> `leaf-value`
    """
    def flatten(prefix: Optional[str],
                values: Any,
                tuple_list: List[Tuple[str, Any]]) -> None:
        if isinstance(values, dict):
            pref = prefix + sep if prefix is not None else ''
            for name, val in values.items():
                flatten(f'{pref}{name}', val, tuple_list)
        else:
            if prefix is None:
                raise ValueError('nested_dict is not a dictionary')
            tuple_list.append((prefix, values))

    tuple_list: List[Tuple[str, Any]] = []
    flatten(key_prefix, nested_dict, tuple_list)
    return dict(tuple_list)


def concat_dicts(dicts: List[Dict[K, V]]) -> Dict[K, List[V]]:
    """Given a list of dictionaries returns a dictionary with lists
    of values for the same keys.

    Resulting dictionary has the same keys as input dictionaries, but
    its values are simply lists of values from these dictionaries for
    corresponding keys.

    Args:
        dicts: dictionaries of values
    Returns:
        lists_dict
    """
    if len(dicts) == 0:
        return {}

    lists_dict: DefaultDict[K, List[V]] = defaultdict(lambda: [])
    for single_dict in dicts:
        for key, val in single_dict.items():
            lists_dict[key].append(val)

    return dict(lists_dict)


def unzip_dict(dictionary: Dict[K, List[Any]]) -> List[Dict[K, Any]]:
    """Given a dictionary having lists/tuples as values returns a list
    of dictionaries with single values.

    Example:
    ```
    unzip_dict(
        dict(a=[1,2,3], b=['A', 'B', 'C', 'D']))
    ==
    [dict(a=1, b='A'), dict(a=2, b='B'), dict(a=3, b='C'), dict(b='D')]
    ```
    """
    out_dicts: List[Dict[K, Any]] = []
    for key, val_list in dictionary.items():
        for ind, val in enumerate(val_list):
            if ind >= len(out_dicts):
                out_dicts.append({})
            out_dicts[ind][key] = val

    return out_dicts


def filter_dict(dictionary: Dict[K, V],
                filter_fn: Callable[[K, V], bool]) -> Dict[K, V]:
    return {
        name: val for name, val in dictionary.items()
        if filter_fn(name, val)
    }
