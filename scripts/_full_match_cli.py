"""Shared, dependency-light helpers for full-match command line tools."""

from __future__ import annotations

import importlib
import inspect
import json
import sys
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


class CliError(RuntimeError):
    """An actionable command-line error."""


def enable_local_imports() -> None:
    """Make the source tree importable without requiring an editable install."""
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))


def require_file(raw: str | Path, label: str) -> Path:
    path = Path(raw).expanduser().resolve()
    if not path.is_file():
        raise CliError(f"{label} does not exist or is not a file: {path}")
    return path


def require_dir(raw: str | Path, label: str) -> Path:
    path = Path(raw).expanduser().resolve()
    if not path.is_dir():
        raise CliError(f"{label} does not exist or is not a directory: {path}")
    return path


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - declared project dependency
        raise CliError("PyYAML is required to read full-match configuration") from exc
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise CliError(f"configuration must contain a YAML mapping: {path}")
    return value


def resolve_api(module_name: str, names: Iterable[str]) -> Callable[..., Any]:
    """Resolve one public entry point from a planned package at execution time."""
    enable_local_imports()
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise CliError(
            f"{module_name} is unavailable. Complete/install the planned package "
            "before running this command."
        ) from exc
    for name in names:
        candidate = getattr(module, name, None)
        if callable(candidate):
            return candidate
    expected = ", ".join(names)
    raise CliError(f"{module_name} does not expose a supported entry point ({expected})")


def call_api(api: Callable[..., Any], **kwargs: Any) -> Any:
    """Call an API while allowing planned implementations to accept a subset."""
    signature = inspect.signature(api)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        return api(**kwargs)
    accepted = {name: value for name, value in kwargs.items() if name in signature.parameters}
    missing = [
        name
        for name, parameter in signature.parameters.items()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        and name not in accepted
    ]
    if missing:
        raise CliError(
            f"{api.__module__}.{api.__name__} requires unsupported arguments: "
            f"{', '.join(missing)}"
        )
    return api(**accepted)


def normalize_result(value: Any) -> dict[str, Any]:
    if value is None:
        return {"status": "PASS"}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        return dict(value.model_dump(mode="json"))
    if hasattr(value, "to_dict"):
        return dict(value.to_dict())
    return {"status": "PASS", "result": str(value)}


def emit(value: Any) -> int:
    payload = normalize_result(value)
    print(json.dumps(payload, indent=2, default=str))
    return 0 if str(payload.get("status", "PASS")).upper() in {"PASS", "OK", "COMPLETE"} else 1


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, default=str) + "\n", encoding="utf-8")


def run_cli(main: Callable[[], int]) -> None:
    try:
        raise SystemExit(main())
    except (CliError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
