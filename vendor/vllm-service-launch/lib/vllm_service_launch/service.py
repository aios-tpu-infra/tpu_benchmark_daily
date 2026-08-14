"""Service lifecycle orchestration around systemd and ephemeral state."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Callable, Iterable, Mapping, NoReturn, Protocol

from .schema import CallerIdentity, CandidateRequest, ServiceRequest
from .state import (
    RuntimeLayout,
    allocate_and_publish,
    claim_request,
    discard_pending,
    locked_request,
    read_request,
    read_runtime,
    reserve_request,
)


class ServiceError(RuntimeError):
    """A systemd-backed service operation failed."""


class SystemdController(Protocol):
    def start(self, service_id: str) -> None: ...

    def stop(self, service_id: str) -> None: ...

    def stop_no_block(self, service_id: str) -> None: ...

    def show(self, service_id: str) -> Mapping[str, str]: ...


class SystemdSubprocessController:
    def __init__(self, executable: str = "/usr/bin/systemctl") -> None:
        self._executable = executable

    def _run(self, arguments: Iterable[str]) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            [self._executable, *arguments],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise ServiceError(detail or "systemctl operation failed")
        return result

    @staticmethod
    def _unit(service_id: str) -> str:
        return f"vllm@{service_id}.service"

    def start(self, service_id: str) -> None:
        self._run(["start", self._unit(service_id)])

    def stop(self, service_id: str) -> None:
        self._run(["stop", self._unit(service_id)])

    def stop_no_block(self, service_id: str) -> None:
        self._run(["--no-block", "stop", self._unit(service_id)])

    def show(self, service_id: str) -> dict[str, str]:
        result = self._run(
            [
                "show",
                self._unit(service_id),
                "--property=ActiveState",
                "--property=SubState",
                "--property=MainPID",
                "--no-pager",
            ]
        )
        properties: dict[str, str] = {}
        for line in result.stdout.splitlines():
            if "=" not in line:
                raise ServiceError("systemctl show returned invalid output")
            key, value = line.split("=", 1)
            if key not in {"ActiveState", "SubState", "MainPID"}:
                raise ServiceError("systemctl show returned unknown property")
            properties[key] = value
        if set(properties) != {"ActiveState", "SubState", "MainPID"}:
            raise ServiceError("systemctl show omitted required properties")
        return properties


def _rollback_pending_submission(
    layout: RuntimeLayout,
    service_id: str,
    request_id: str,
    *,
    systemd: SystemdController,
    stop_unit: bool,
) -> list[str]:
    errors: list[str] = []
    if stop_unit:
        try:
            systemd.stop(service_id)
        except Exception as exc:  # best-effort cleanup preserves the original error
            errors.append(f"cannot stop unit: {exc}")
    try:
        discard_pending(
            layout,
            service_id,
            request_id=request_id,
        )
    except Exception as exc:  # best-effort cleanup preserves the original error
        errors.append(f"cannot discard pending request: {exc}")
    return errors


def _raise_submission_error(
    exc: Exception,
    cleanup_errors: list[str],
) -> NoReturn:
    if cleanup_errors:
        detail = "; ".join(cleanup_errors)
        raise ServiceError(f"{exc}; submission cleanup failed: {detail}") from exc
    raise exc


def submit_request(
    layout: RuntimeLayout,
    candidate: CandidateRequest,
    caller: CallerIdentity,
    *,
    systemd: SystemdController,
    request_id_factory: Callable[[], str],
    claim_timeout: float = 5.0,
) -> ServiceRequest:
    request_id = request_id_factory()
    request = reserve_request(
        layout,
        candidate,
        caller,
        request_id=request_id,
    )
    try:
        systemd.start(candidate.service_id)
    except Exception as exc:
        cleanup_errors = _rollback_pending_submission(
            layout,
            candidate.service_id,
            request_id,
            systemd=systemd,
            stop_unit=False,
        )
        _raise_submission_error(exc, cleanup_errors)

    deadline = time.monotonic() + claim_timeout
    stop_unit = True
    try:
        while True:
            current = read_request(layout, candidate.service_id)
            if current.request_id != request.request_id:
                raise ServiceError("request ownership changed during submit")
            if current.state != "pending":
                return current

            properties = systemd.show(candidate.service_id)
            if properties["ActiveState"] in {"inactive", "failed"}:
                stop_unit = False
                raise ServiceError("systemd exited before claiming the request")
            if time.monotonic() >= deadline:
                raise ServiceError("systemd did not claim the request in time")
            time.sleep(0.05)
    except Exception as exc:
        cleanup_errors = _rollback_pending_submission(
            layout,
            candidate.service_id,
            request_id,
            systemd=systemd,
            stop_unit=stop_unit,
        )
        _raise_submission_error(exc, cleanup_errors)


def run_service(
    layout: RuntimeLayout,
    service_id: str,
    *,
    invocation_id: str,
    pid: int,
    dropper: Callable[[CallerIdentity], None],
    environment_builder: Callable[[ServiceRequest], dict[str, str]],
    change_directory: Callable[[Path], None],
    executor: Callable[[str, list[str], Mapping[str, str]], None],
    output_redirector: Callable[[Path], None],
) -> None:
    request = claim_request(
        layout,
        service_id,
        invocation_id=invocation_id,
    )
    runtime = allocate_and_publish(layout, request, pid=pid)
    dropper(request.caller)
    environment = environment_builder(request)
    change_directory(request.candidate.working_directory)
    if request.candidate.output is not None:
        output_redirector(request.candidate.output)
    if request.candidate.runtime == "vllm":
        executable = str(request.candidate.environment.vllm_executable)
        argv = [
            executable,
            *request.candidate.vllm_argv,
            "--host",
            runtime.listen_host,
            "--port",
            str(runtime.port),
        ]
    else:
        executable = request.candidate.vllm_argv[0]
        argv = list(request.candidate.vllm_argv)
        environment["DS_LLM_PROMETHEUS_HOST"] = runtime.listen_host
        environment["DS_LLM_PROMETHEUS_PORT"] = str(runtime.port)
    executor(executable, argv, environment)
    raise ServiceError("service executor returned without replacing the process")


def redirect_output(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        0o600,
    )
    try:
        os.fchmod(descriptor, 0o600)
        os.dup2(descriptor, 1)
        os.dup2(descriptor, 2)
    finally:
        if descriptor not in {1, 2}:
            os.close(descriptor)


def _authorize(
    request: ServiceRequest,
    *,
    requester_uid: int,
    administrator: bool,
) -> None:
    if not administrator and request.caller.uid != requester_uid:
        raise ServiceError("caller does not own this service")


def service_status(
    layout: RuntimeLayout,
    service_id: str,
    *,
    requester_uid: int,
    administrator: bool,
    systemd: SystemdController,
) -> dict[str, object]:
    request = read_request(layout, service_id)
    _authorize(
        request,
        requester_uid=requester_uid,
        administrator=administrator,
    )
    payload: dict[str, object] = {
        "service_id": service_id,
        "request_id": request.request_id,
        "state": request.state,
        "runtime": request.candidate.runtime,
        "systemd": dict(systemd.show(service_id)),
    }
    runtime = read_runtime(layout, service_id)
    if runtime is not None:
        payload.update(
            {
                "pid": runtime.pid,
                "listen_host": runtime.listen_host,
                "scrape_host": runtime.scrape_host,
                "port": runtime.port,
            }
        )
    return payload


def stop_service(
    layout: RuntimeLayout,
    service_id: str,
    *,
    requester_uid: int,
    administrator: bool,
    systemd: SystemdController,
    expected_request_id: str | None = None,
) -> None:
    with locked_request(layout, service_id) as request:
        _authorize(
            request,
            requester_uid=requester_uid,
            administrator=administrator,
        )
        if (
            expected_request_id is not None
            and request.request_id != expected_request_id
        ):
            raise ServiceError("request ownership changed before stop")
        systemd.stop_no_block(service_id)
