"""Structural test: CLI/reader parameter alignment.

Every parameter of each reader function must be exposed by its CLI wrapper, and
every wrapper parameter (beyond ``input``/``output``) must correspond to a reader
parameter.  Both directions are checked as a symmetric difference.

Runs automatically in CI on every PR to catch drift between readers and
__main__.py before it reaches users.

What this test does NOT check
------------------------------
- **Types**: a reader may declare ``imread_kwargs: Mapping[str, Any]`` while the
  CLI accepts a JSON string (``str``).  The types will not match, and this test
  will not catch that mismatch.
- **Default values**: a reader default and the corresponding ``@click.option``
  default can silently diverge (e.g. ``n_jobs=None`` vs ``default=1``).

Both of these require manual or AI-assisted review.  Run the following command
to ask Claude Code to audit type and default alignment for a specific reader::

    claude "In src/spatialdata_io/__main__.py, compare every @click.option
    default and type annotation in <wrapper_name> against the corresponding
    parameter in the reader function in src/spatialdata_io/readers/<reader>.py.
    List any mismatches in default values or types."

When this test turns red, fix it by adding (or removing) the relevant
``@click.option`` and parameter in ``__main__.py``.  If a reader parameter
should intentionally stay hidden (e.g. deprecated), add it to
``_READER_EXCEPTIONS`` with a comment explaining why.
"""

import inspect
from typing import Any

import pytest


def _reader_case(module_path: str, reader_name: str, wrapper_name: str) -> Any:
    return pytest.param(module_path, reader_name, wrapper_name, marks=getattr(pytest.mark, reader_name), id=reader_name)


# (reader_module_path, reader_func_name, cli_wrapper_func_name)
_READERS = [
    _reader_case("spatialdata_io.readers.codex", "codex", "codex_wrapper"),
    _reader_case("spatialdata_io.readers.cosmx", "cosmx", "cosmx_wrapper"),
    _reader_case("spatialdata_io.readers.curio", "curio", "curio_wrapper"),
    _reader_case("spatialdata_io.readers.dbit", "dbit", "dbit_wrapper"),
    _reader_case("spatialdata_io.readers.iss", "iss", "iss_wrapper"),
    _reader_case("spatialdata_io.readers.macsima", "macsima", "macsima_wrapper"),
    _reader_case("spatialdata_io.readers.mcmicro", "mcmicro", "mcmicro_wrapper"),
    _reader_case("spatialdata_io.readers.merscope", "merscope", "merscope_wrapper"),
    _reader_case("spatialdata_io.readers.seqfish", "seqfish", "seqfish_wrapper"),
    _reader_case("spatialdata_io.readers.steinbock", "steinbock", "steinbock_wrapper"),
    _reader_case("spatialdata_io.readers.stereoseq", "stereoseq", "stereoseq_wrapper"),
    _reader_case("spatialdata_io.readers.visium", "visium", "visium_wrapper"),
    _reader_case("spatialdata_io.readers.visium_hd", "visium_hd", "visium_hd_wrapper"),
    _reader_case("spatialdata_io.readers.xenium", "xenium", "xenium_wrapper"),
]

# Parameters to skip in the reader (first positional path arg, and **kwargs catch-alls)
_READER_SKIP = {"path"}
# Parameters to skip in the wrapper (always present, not reader-specific)
_WRAPPER_SKIP = {"input", "output"}

# Intentionally unexposed reader parameters: {reader_func_name: {param, ...}}
# Document *why* each entry exists so future contributors can re-evaluate.
_READER_EXCEPTIONS: dict[str, set[str]] = {
    # n_jobs is deprecated in the xenium reader (kept for backward compat but
    # has no effect); exposing it in the CLI would mislead users.
    "xenium": {"n_jobs"},
}


def _reader_params(module_path: str, func_name: str) -> set[str]:
    import importlib

    module = importlib.import_module(module_path)
    func = getattr(module, func_name)
    sig = inspect.signature(func)
    return {
        name
        for name, param in sig.parameters.items()
        if param.kind not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
        and name not in _READER_SKIP
    }


def _wrapper_params(func_name: str) -> set[str]:
    import importlib

    module = importlib.import_module("spatialdata_io.__main__")
    obj = getattr(module, func_name)
    # @cli.command replaces the function with a click.Command; the Python
    # function lives in .callback.
    func = obj.callback if hasattr(obj, "callback") else obj
    sig = inspect.signature(func)
    return {name for name in sig.parameters if name not in _WRAPPER_SKIP}


@pytest.mark.parametrize("module_path,reader_name,wrapper_name", _READERS)
def test_cli_exposes_all_reader_params(module_path: str, reader_name: str, wrapper_name: str) -> None:
    """CLI wrapper and reader must have exactly the same parameter names (symmetric)."""
    reader = _reader_params(module_path, reader_name)
    wrapper = _wrapper_params(wrapper_name)
    exceptions = _READER_EXCEPTIONS.get(reader_name, set())

    # Reader params absent from the CLI — the wrapper is missing options.
    missing = reader - wrapper - exceptions
    assert not missing, (
        f"{wrapper_name} is missing CLI parameters for reader '{reader_name}': {sorted(missing)}\n"
        f"Add @click.option and a matching parameter to {wrapper_name} in __main__.py.\n"
        f"If the parameter should intentionally be hidden, add it to _READER_EXCEPTIONS['{reader_name}']."
    )

    # Wrapper params absent from the reader — the CLI has stale/extra options.
    extra = wrapper - reader - exceptions
    assert not extra, (
        f"{wrapper_name} has CLI parameters not present in reader '{reader_name}': {sorted(extra)}\n"
        f"Remove the corresponding @click.option from {wrapper_name} in __main__.py,\n"
        f"or add the parameter to the reader if it was omitted by mistake."
    )
