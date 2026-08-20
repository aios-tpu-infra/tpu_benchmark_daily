"""Strict wire contracts for the stateless vLLM service launcher."""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

SCHEMA_VERSION = 1
SERVICE_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9_.-]{0,127}")
OPAQUE_ID_PATTERN = re.compile(r"[0-9a-f]{32}")
ENVIRONMENT_KEYS = frozenset(
    {"kind", "executable", "prefix", "project", "vllm_executable"}
)


class ContractError(ValueError):
    """An input does not satisfy the launcher wire contract."""


def _strict_object(
    value: object,
    *,
    name: str,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{name} must be an object")
    keys = frozenset(value)
    missing = required - keys
    unknown = keys - required - optional
    if missing:
        raise ContractError(f"{name} is missing required fields")
    if unknown:
        raise ContractError(f"{name} contains unknown fields")
    return value


def _required_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{name} must be a non-empty string")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ContractError(f"{name} must not contain control characters")
    return value


def _absolute_path(value: object, *, name: str) -> Path:
    path = Path(_required_string(value, name=name))
    if not path.is_absolute():
        raise ContractError(f"{name} must be an absolute path")
    return path


def _port(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{name} must be an integer")
    if not 1 <= value <= 65535:
        raise ContractError(f"{name} must be in 1:65535")
    return value


def _require_schema_version(value: object, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != SCHEMA_VERSION:
        raise ContractError(f"unsupported {name} schema_version")


@dataclass(frozen=True)
class EnvironmentSpec:
    kind: Literal["conda", "uv"]
    executable: Path
    prefix: Path
    vllm_executable: Path
    project: Path | None

    @classmethod
    def from_dict(cls, value: object) -> EnvironmentSpec:
        payload = _strict_object(
            value,
            name="environment",
            required=frozenset({"kind", "executable", "prefix", "vllm_executable"}),
            optional=frozenset({"project"}),
        )
        kind = payload["kind"]
        if kind not in {"conda", "uv"}:
            raise ContractError("environment.kind must be conda or uv")
        project_value = payload.get("project")
        if kind == "conda" and project_value is not None:
            raise ContractError("Conda environment must not define project")
        if kind == "uv" and project_value is None:
            raise ContractError("uv environment requires project")
        return cls(
            kind=kind,
            executable=_absolute_path(
                payload["executable"], name="environment.executable"
            ),
            prefix=_absolute_path(payload["prefix"], name="environment.prefix"),
            vllm_executable=_absolute_path(
                payload["vllm_executable"],
                name="environment.vllm_executable",
            ),
            project=(
                _absolute_path(project_value, name="environment.project")
                if project_value is not None
                else None
            ),
        )

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "kind": self.kind,
            "executable": str(self.executable),
            "prefix": str(self.prefix),
            "vllm_executable": str(self.vllm_executable),
        }
        if self.project is not None:
            payload["project"] = str(self.project)
        return payload


@dataclass(frozen=True)
class FixedPortPolicy:
    mode: Literal["fixed"]
    port: int

    @classmethod
    def from_dict(cls, value: object) -> FixedPortPolicy:
        payload = _strict_object(
            value,
            name="port_policy",
            required=frozenset({"mode", "port"}),
        )
        if payload["mode"] != "fixed":
            raise ContractError("port_policy.mode must be fixed")
        return cls(mode="fixed", port=_port(payload["port"], name="port"))

    def to_dict(self) -> dict[str, object]:
        return {"mode": self.mode, "port": self.port}


@dataclass(frozen=True)
class AutoPortPolicy:
    mode: Literal["auto"]
    range_start: int
    range_end: int

    @classmethod
    def from_dict(cls, value: object) -> AutoPortPolicy:
        payload = _strict_object(
            value,
            name="port_policy",
            required=frozenset({"mode", "range_start", "range_end"}),
        )
        if payload["mode"] != "auto":
            raise ContractError("port_policy.mode must be auto")
        range_start = _port(payload["range_start"], name="range_start")
        range_end = _port(payload["range_end"], name="range_end")
        if range_start > range_end:
            raise ContractError("range_start must not exceed range_end")
        return cls(
            mode="auto",
            range_start=range_start,
            range_end=range_end,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "range_start": self.range_start,
            "range_end": self.range_end,
        }


PortPolicy = FixedPortPolicy | AutoPortPolicy
RuntimeKind = Literal["vllm", "dashllm"]


def _validate_vllm_argv(argv: tuple[str, ...]) -> None:
    for argument in argv:
        if argument in {"--host", "--port"} or argument.startswith(
            ("--host=", "--port=")
        ):
            raise ContractError("vLLM argv must not define host or port")

    if argv[0] != "serve":
        return
    if any(argument in {"--headless", "--grpc"} for argument in argv):
        raise ContractError("vLLM serve must expose HTTP /metrics")

    for index, argument in enumerate(argv):
        if argument == "--api-server-count":
            if index + 1 == len(argv):
                continue
            value = argv[index + 1]
        elif argument.startswith("--api-server-count="):
            value = argument.partition("=")[2]
        else:
            continue
        try:
            api_server_count = int(value)
        except ValueError:
            continue
        if api_server_count < 1:
            raise ContractError("vLLM serve must expose HTTP /metrics")


def _parse_port_policy(value: object) -> PortPolicy:
    if not isinstance(value, dict):
        raise ContractError("port_policy must be an object")
    mode = value.get("mode")
    if mode == "fixed":
        return FixedPortPolicy.from_dict(value)
    if mode == "auto":
        return AutoPortPolicy.from_dict(value)
    raise ContractError("port_policy.mode must be fixed or auto")


@dataclass(frozen=True)
class CandidateRequest:
    service_id: str
    role: Literal["prefill", "decode", "mixed"]
    model_alias: str
    environment: EnvironmentSpec
    env_files: tuple[Path, ...]
    working_directory: Path
    listen_host: str
    port_policy: PortPolicy
    runtime: RuntimeKind
    output: Path | None
    vllm_argv: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: object) -> CandidateRequest:
        payload = _strict_object(
            value,
            name="candidate request",
            required=frozenset(
                {
                    "schema_version",
                    "service_id",
                    "role",
                    "model_alias",
                    "environment",
                    "env_files",
                    "working_directory",
                    "listen_host",
                    "port_policy",
                    "vllm_argv",
                }
            ),
            optional=frozenset({"runtime", "output"}),
        )
        _require_schema_version(
            payload["schema_version"],
            name="candidate",
        )

        service_id = _required_string(payload["service_id"], name="service_id")
        if SERVICE_ID_PATTERN.fullmatch(service_id) is None:
            raise ContractError("invalid service_id")

        role = payload["role"]
        if role not in {"prefill", "decode", "mixed"}:
            raise ContractError("invalid role")

        env_files_value = payload["env_files"]
        if not isinstance(env_files_value, list):
            raise ContractError("env_files must be an array")
        env_files = tuple(
            _absolute_path(path, name="env_files item") for path in env_files_value
        )

        listen_host = _required_string(payload["listen_host"], name="listen_host")
        try:
            ipaddress.ip_address(listen_host)
        except ValueError as exc:
            raise ContractError("listen_host must be an IP literal") from exc

        runtime = payload.get("runtime", "vllm")
        if runtime not in {"vllm", "dashllm"}:
            raise ContractError("runtime must be vllm or dashllm")

        output_value = payload.get("output")
        output = (
            _absolute_path(output_value, name="output")
            if output_value is not None
            else None
        )

        argv_value = payload["vllm_argv"]
        if not isinstance(argv_value, list) or not argv_value:
            raise ContractError("vllm_argv must be a non-empty array")
        argv = tuple(
            _required_string(argument, name="vllm_argv item") for argument in argv_value
        )
        if runtime == "vllm":
            _validate_vllm_argv(argv)
        elif not Path(argv[0]).is_absolute():
            raise ContractError("DashLLM executable must be absolute")

        return cls(
            service_id=service_id,
            role=role,
            model_alias=_required_string(payload["model_alias"], name="model_alias"),
            environment=EnvironmentSpec.from_dict(payload["environment"]),
            env_files=env_files,
            working_directory=_absolute_path(
                payload["working_directory"], name="working_directory"
            ),
            listen_host=listen_host,
            port_policy=_parse_port_policy(payload["port_policy"]),
            runtime=runtime,
            output=output,
            vllm_argv=argv,
        )

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "service_id": self.service_id,
            "role": self.role,
            "model_alias": self.model_alias,
            "environment": self.environment.to_dict(),
            "env_files": [str(path) for path in self.env_files],
            "working_directory": str(self.working_directory),
            "listen_host": self.listen_host,
            "port_policy": self.port_policy.to_dict(),
            "runtime": self.runtime,
            "vllm_argv": list(self.vllm_argv),
        }
        if self.output is not None:
            payload["output"] = str(self.output)
        return payload


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractError(f"{name} must be a non-negative integer")
    return value


def _boolean(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ContractError(f"{name} must be a boolean")
    return value


def _opaque_id(value: object, *, name: str) -> str:
    opaque_id = _required_string(value, name=name)
    if OPAQUE_ID_PATTERN.fullmatch(opaque_id) is None:
        raise ContractError(f"{name} must be 32 lowercase hex characters")
    return opaque_id


@dataclass(frozen=True)
class ServiceRequest:
    request_id: str
    candidate: CandidateRequest
    starter: ProcessIdentity
    supervisor: ProcessIdentity | None
    cancellation_requested: bool

    @property
    def service_id(self) -> str:
        return self.candidate.service_id

    @classmethod
    def from_dict(cls, value: object) -> ServiceRequest:
        payload = _strict_object(
            value,
            name="service request",
            required=frozenset(
                {
                    "schema_version",
                    "request_id",
                    "candidate",
                    "starter",
                    "cancellation_requested",
                }
            ),
            optional=frozenset({"supervisor"}),
        )
        _require_schema_version(payload["schema_version"], name="service request")
        return cls(
            request_id=_opaque_id(payload["request_id"], name="request_id"),
            candidate=CandidateRequest.from_dict(payload["candidate"]),
            starter=ProcessIdentity.from_dict(payload["starter"], name="starter"),
            supervisor=(
                ProcessIdentity.from_dict(
                    payload["supervisor"],
                    name="supervisor",
                )
                if payload.get("supervisor") is not None
                else None
            ),
            cancellation_requested=_boolean(
                payload["cancellation_requested"],
                name="cancellation_requested",
            ),
        )

    def to_dict(self) -> dict[str, object]:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "request_id": self.request_id,
            "candidate": self.candidate.to_dict(),
            "starter": self.starter.to_dict(),
            "cancellation_requested": self.cancellation_requested,
        }
        if self.supervisor is not None:
            payload["supervisor"] = self.supervisor.to_dict()
        return payload


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    start_time: int

    @classmethod
    def from_dict(cls, value: object, *, name: str) -> ProcessIdentity:
        payload = _strict_object(
            value,
            name=name,
            required=frozenset({"pid", "start_time"}),
        )
        pid = payload["pid"]
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            raise ContractError(f"{name}.pid must be a positive integer")
        return cls(
            pid=pid,
            start_time=_nonnegative_integer(
                payload["start_time"],
                name=f"{name}.start_time",
            ),
        )

    def to_dict(self) -> dict[str, int]:
        return {"pid": self.pid, "start_time": self.start_time}


@dataclass(frozen=True)
class RuntimeState:
    request_id: str
    service_id: str
    supervisor: ProcessIdentity
    server: ProcessIdentity
    server_pgid: int
    server_session_id: int
    listen_host: str
    scrape_host: str
    port: int

    @classmethod
    def from_dict(cls, value: object) -> RuntimeState:
        payload = _strict_object(
            value,
            name="runtime",
            required=frozenset(
                {
                    "schema_version",
                    "request_id",
                    "service_id",
                    "supervisor",
                    "server",
                    "server_pgid",
                    "server_session_id",
                    "listen_host",
                    "scrape_host",
                    "port",
                }
            ),
        )
        _require_schema_version(payload["schema_version"], name="runtime")
        service_id = _required_string(payload["service_id"], name="service_id")
        if SERVICE_ID_PATTERN.fullmatch(service_id) is None:
            raise ContractError("invalid service_id")
        server_pgid = payload["server_pgid"]
        if (
            isinstance(server_pgid, bool)
            or not isinstance(server_pgid, int)
            or server_pgid <= 0
        ):
            raise ContractError("server_pgid must be a positive integer")
        server_session_id = payload["server_session_id"]
        if (
            isinstance(server_session_id, bool)
            or not isinstance(server_session_id, int)
            or server_session_id <= 0
        ):
            raise ContractError("server_session_id must be a positive integer")
        listen_host = _required_string(payload["listen_host"], name="listen_host")
        scrape_host = _required_string(payload["scrape_host"], name="scrape_host")
        try:
            ipaddress.ip_address(listen_host)
            ipaddress.ip_address(scrape_host)
        except ValueError as exc:
            raise ContractError("runtime hosts must be IP literals") from exc
        return cls(
            request_id=_opaque_id(payload["request_id"], name="request_id"),
            service_id=service_id,
            supervisor=ProcessIdentity.from_dict(
                payload["supervisor"], name="supervisor"
            ),
            server=ProcessIdentity.from_dict(payload["server"], name="server"),
            server_pgid=server_pgid,
            server_session_id=server_session_id,
            listen_host=listen_host,
            scrape_host=scrape_host,
            port=_port(payload["port"], name="port"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "request_id": self.request_id,
            "service_id": self.service_id,
            "supervisor": self.supervisor.to_dict(),
            "server": self.server.to_dict(),
            "server_pgid": self.server_pgid,
            "server_session_id": self.server_session_id,
            "listen_host": self.listen_host,
            "scrape_host": self.scrape_host,
            "port": self.port,
        }
