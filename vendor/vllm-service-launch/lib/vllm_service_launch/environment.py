"""Conda, uv, and runtime environment handling."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Mapping

from .schema import EnvironmentSpec, ServiceRequest

ENV_KEY_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
RESERVED_ENVIRONMENT_KEYS = frozenset(
    {
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "PATH",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
        "VLLM_HOST",
        "VLLM_PORT",
        "VLLM_SERVICE_ID",
        "VLLM_ROLE",
        "VLLM_MODEL_ALIAS",
        "VLLM_LAUNCH_SCRIPT",
        "DS_LLM_PROMETHEUS_HOST",
        "DS_LLM_PROMETHEUS_PORT",
    }
)


class RuntimeEnvironmentError(RuntimeError):
    """A requested Python or process environment is invalid."""


def _run_json(command: list[str]) -> object:
    result = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeEnvironmentError(f"environment command failed: {command[0]}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeEnvironmentError(
            f"environment command returned invalid JSON: {command[0]}"
        ) from exc


def _find_executable(name: str) -> Path:
    executable = shutil.which(name)
    if executable is None:
        raise RuntimeEnvironmentError(f"{name} is not available in PATH")
    return Path(executable).resolve()


def _executable_file(path: Path, *, name: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise RuntimeEnvironmentError(f"{name} is not executable")
    return resolved


def resolve_conda_environment(name_or_prefix: str) -> EnvironmentSpec:
    if not name_or_prefix:
        raise RuntimeEnvironmentError("Conda environment must not be empty")
    conda = _find_executable("conda")
    payload = _run_json([str(conda), "env", "list", "--json"])
    if not isinstance(payload, dict) or "envs" not in payload:
        raise RuntimeEnvironmentError("Conda environment list has invalid schema")
    envs = payload["envs"]
    if not isinstance(envs, list) or not all(isinstance(value, str) for value in envs):
        raise RuntimeEnvironmentError("Conda environment list has invalid entries")

    requested = Path(name_or_prefix)
    if requested.is_absolute():
        requested_path = requested.resolve()
        matches = [
            Path(value).resolve()
            for value in envs
            if Path(value).resolve() == requested_path
        ]
    else:
        matches = [
            Path(value).resolve()
            for value in envs
            if Path(value).name == name_or_prefix
        ]
    if len(matches) != 1:
        raise RuntimeEnvironmentError(
            "Conda environment must resolve to exactly one prefix"
        )

    prefix = matches[0]
    vllm = _executable_file(prefix / "bin" / "vllm", name="vllm")
    return EnvironmentSpec(
        kind="conda",
        executable=conda,
        prefix=prefix,
        vllm_executable=vllm,
        project=None,
    )


def resolve_uv_environment(project: Path) -> EnvironmentSpec:
    project = project.resolve()
    if not project.is_dir():
        raise RuntimeEnvironmentError("uv project must be an existing directory")
    uv = _find_executable("uv")
    resolver = (
        "import json, shutil, sys; "
        "vllm = shutil.which('vllm'); "
        "print(json.dumps({'prefix': sys.prefix, "
        "'vllm_executable': vllm}))"
    )
    payload = _run_json(
        [
            str(uv),
            "run",
            "--project",
            str(project),
            "--no-sync",
            "python",
            "-c",
            resolver,
        ]
    )
    if not isinstance(payload, dict) or set(payload) != {
        "prefix",
        "vllm_executable",
    }:
        raise RuntimeEnvironmentError("uv environment result has invalid schema")
    prefix_value = payload["prefix"]
    vllm_value = payload["vllm_executable"]
    if not isinstance(prefix_value, str) or not isinstance(vllm_value, str):
        raise RuntimeEnvironmentError("uv environment paths must be strings")
    prefix = Path(prefix_value).resolve()
    if not prefix.is_dir():
        raise RuntimeEnvironmentError("uv environment prefix does not exist")
    vllm = _executable_file(Path(vllm_value), name="vllm")
    return EnvironmentSpec(
        kind="uv",
        executable=uv,
        prefix=prefix,
        vllm_executable=vllm,
        project=project,
    )


def parse_env_files(
    paths: Iterable[Path],
    base_environment: Mapping[str, str],
) -> dict[str, str]:
    environment = dict(base_environment)
    for path in paths:
        if not path.is_file():
            raise RuntimeEnvironmentError(f"env file does not exist: {path}")
        seen: set[str] = set()
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError) as exc:
            raise RuntimeEnvironmentError(f"cannot read env file: {path}") from exc
        for line in lines:
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                raise RuntimeEnvironmentError(f"invalid env assignment in {path}")
            key, value = line.split("=", 1)
            if ENV_KEY_PATTERN.fullmatch(key) is None:
                raise RuntimeEnvironmentError(f"invalid env key in {path}")
            if key in seen:
                raise RuntimeEnvironmentError(f"duplicate env key in {path}")
            if key in RESERVED_ENVIRONMENT_KEYS:
                raise RuntimeEnvironmentError(f"reserved env key in {path}")
            if "\0" in value:
                raise RuntimeEnvironmentError(f"env value contains NUL in {path}")
            seen.add(key)
            environment[key] = value
    return environment


def _captured_environment(
    command: list[str],
    base_environment: Mapping[str, str],
) -> dict[str, str]:
    result = subprocess.run(
        command,
        env=dict(base_environment),
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeEnvironmentError(f"environment activation failed: {command[0]}")
    try:
        decoded = result.stdout.decode("utf-8")
    except UnicodeError as exc:
        raise RuntimeEnvironmentError(
            "activated environment is not valid UTF-8"
        ) from exc
    environment: dict[str, str] = {}
    for assignment in decoded.split("\0"):
        if not assignment:
            continue
        if "=" not in assignment:
            raise RuntimeEnvironmentError(
                "activated environment contains an invalid assignment"
            )
        key, value = assignment.split("=", 1)
        if ENV_KEY_PATTERN.fullmatch(key) is None or key in environment:
            raise RuntimeEnvironmentError(
                "activated environment contains an invalid key"
            )
        environment[key] = value
    if "PATH" not in environment:
        raise RuntimeEnvironmentError("activated environment does not define PATH")
    return environment


def build_runtime_environment(
    request: ServiceRequest,
) -> dict[str, str]:
    caller = request.caller
    base_environment = {
        "HOME": str(caller.home),
        "USER": caller.name,
        "LOGNAME": caller.name,
        "SHELL": str(caller.shell),
        "PATH": "/usr/local/bin:/usr/bin:/bin",
    }
    specification = request.candidate.environment
    if specification.kind == "conda":
        command = [
            str(specification.executable),
            "run",
            "-p",
            str(specification.prefix),
            "--no-capture-output",
            "/usr/bin/env",
            "-0",
        ]
    elif specification.kind == "uv":
        if specification.project is None:
            raise RuntimeEnvironmentError("uv environment requires project")
        command = [
            str(specification.executable),
            "run",
            "--project",
            str(specification.project),
            "--no-sync",
            "/usr/bin/env",
            "-0",
        ]
    else:
        raise RuntimeEnvironmentError("unsupported environment kind")

    activated = _captured_environment(command, base_environment)
    if specification.kind == "conda" and activated.get("CONDA_PREFIX") != str(
        specification.prefix
    ):
        raise RuntimeEnvironmentError("Conda activated the wrong prefix")
    activated.update(
        {
            "HOME": base_environment["HOME"],
            "USER": base_environment["USER"],
            "LOGNAME": base_environment["LOGNAME"],
            "SHELL": base_environment["SHELL"],
        }
    )
    return parse_env_files(request.candidate.env_files, activated)
