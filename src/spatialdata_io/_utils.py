from __future__ import annotations

import functools
import warnings
from collections.abc import Callable
from typing import Any, TypeVar

RT = TypeVar("RT")


# these two functions should be removed and imported from spatialdata._utils once the multi_table branch, which
# introduces them, is merged
def deprecation_alias(**aliases: str) -> Callable[[Callable[..., RT]], Callable[..., RT]]:
    """
    Decorate a function to warn user of use of arguments set for deprecation.

    Parameters
    ----------
    aliases
        Deprecation argument aliases to be mapped to the new arguments.

    Returns
    -------
    A decorator that can be used to mark an argument for deprecation and substituting it with the new argument.

    Raises
    ------
    TypeError
        If the provided aliases are not of string type.

    Example
    -------
    Assuming we have an argument 'table' set for deprecation and we want to warn the user and substitute with 'tables':

    .. code-block:: python

        @deprecation_alias(table="tables")
        def my_function(tables):
            pass
    """

    def deprecation_decorator(f: Callable[..., RT]) -> Callable[..., RT]:
        @functools.wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> RT:
            class_name = f.__qualname__
            rename_kwargs(f.__name__, kwargs, aliases, class_name)
            return f(*args, **kwargs)

        return wrapper

    return deprecation_decorator


def rename_kwargs(func_name: str, kwargs: dict[str, Any], aliases: dict[str, str], class_name: None | str) -> None:
    """Rename function arguments set for deprecation and gives warning in case of usage of these arguments."""
    for alias, new in aliases.items():
        if alias in kwargs:
            class_name = class_name + "." if class_name else ""
            if new in kwargs:
                raise TypeError(
                    f"{class_name}{func_name} received both {alias} and {new} as arguments!"
                    f" {alias} is being deprecated, please only use {new} instead."
                )
            warnings.warn(
                message=(
                    f"`{alias}` is being deprecated as an argument to `{class_name}{func_name}` in SpatialData "
                    f"version 0.1, switch to `{new}` instead."
                ),
                category=DeprecationWarning,
                stacklevel=3,
            )
            kwargs[new] = kwargs.pop(alias)
