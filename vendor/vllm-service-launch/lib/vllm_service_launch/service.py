"""Per-service supervisor lifecycle without an external init system."""

from __future__ import annotations

import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time
from typing import Mapping
import uuid

from .endpoint import allocate_endpoint
from .environment import build_runtime_environment
from .process import (
    identity_is_alive,
    process_group_is_alive,
    read_process,
    signal_process_group,
    terminate_process_group,
    unowned_process_group_is_observed,
)
from .schema import CandidateRequest, ProcessIdentity, RuntimeState, ServiceRequest
from .state import (
    RuntimeLayout,
    StateError,
    attach_supervisor,
    cleanup_service,
    locked_registry,
    publish_runtime,
    read_request,
    read_runtime,
    reserve_request,
    reserved_ports,
    request_cancellation,
)


START_ACK_TIMEOUT_SECONDS = 30.0
STOP_TIMEOUT_SECONDS = 120.0
START_CANCEL_TIMEOUT_SECONDS = 5.0


class ServiceError(RuntimeError):
    """A supervised service operation failed."""


def _log(service_id: str, message: str) -> None:
    print(
        f"vllm-service-launch[{service_id}] supervisor_pid={os.getpid()} {message}",
        flush=True,
    )


def _write_ack(descriptor: int, payload: Mapping[str, object]) -> None:
    encoded = (json.dumps(dict(payload), sort_keys=True) + "\n").encode("utf-8")
    while encoded:
        written = os.write(descriptor, encoded)
        encoded = encoded[written:]


def _read_ack(descriptor: int, timeout: float) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    payload = bytearray()
    while b"\n" not in payload:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ServiceError("supervisor did not acknowledge startup in time")
        readable, _, _ = select.select([descriptor], [], [], remaining)
        if not readable:
            raise ServiceError("supervisor did not acknowledge startup in time")
        chunk = os.read(descriptor, 4096)
        if not chunk:
            raise ServiceError("supervisor exited before acknowledging startup")
        payload.extend(chunk)
        if len(payload) > 65536:
            raise ServiceError("supervisor startup acknowledgement is too large")
    line, _, trailing = bytes(payload).partition(b"\n")
    if trailing:
        raise ServiceError("supervisor sent trailing startup acknowledgement data")
    try:
        decoded = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ServiceError("supervisor sent an invalid startup acknowledgement") from exc
    if not isinstance(decoded, dict):
        raise ServiceError("supervisor startup acknowledgement must be an object")
    return decoded


def _request(candidate: CandidateRequest) -> ServiceRequest:
    return ServiceRequest.from_dict(
        {
            "schema_version": 1,
            "request_id": uuid.uuid4().hex,
            "candidate": candidate.to_dict(),
            "starter": _identity(os.getpid()).to_dict(),
            "cancellation_requested": False,
        }
    )


def _runtime_group_is_alive(runtime: RuntimeState) -> bool:
    anchor = runtime.supervisor if identity_is_alive(runtime.supervisor) else None
    return process_group_is_alive(
        runtime.server,
        runtime.server_pgid,
        runtime.server_session_id,
        anchor=anchor,
    )


def _raise_if_ambiguous_orphan(runtime: RuntimeState) -> None:
    if identity_is_alive(runtime.supervisor):
        return
    if unowned_process_group_is_observed(
        runtime.server,
        runtime.server_pgid,
        runtime.server_session_id,
    ):
        raise ServiceError(
            "server leader is gone and no live supervisor can prove process "
            "group ownership; refusing to signal numeric PGID"
        )


def _reconcile_before_start(layout: RuntimeLayout, service_id: str) -> None:
    if not layout.request_path(service_id).exists():
        return
    request = read_request(layout, service_id)
    runtime = read_runtime(layout, service_id)
    if runtime is None:
        pending_owner_alive = identity_is_alive(request.starter) or (
            request.supervisor is not None
            and identity_is_alive(request.supervisor)
        )
        if pending_owner_alive:
            raise ServiceError(f"service is already reserved: {service_id}")
        if not request.cancellation_requested:
            raise ServiceError(
                f"unclaimed service reservation requires explicit stop: {service_id}"
            )
        cleanup_service(
            layout,
            service_id,
            expected_request_id=request.request_id,
        )
        return
    if identity_is_alive(runtime.supervisor) or _runtime_group_is_alive(runtime):
        raise ServiceError(f"service is already running: {service_id}")
    _raise_if_ambiguous_orphan(runtime)
    cleanup_service(
        layout,
        service_id,
        expected_request_id=request.request_id,
    )


def start_service(
    layout: RuntimeLayout,
    candidate: CandidateRequest,
    *,
    launcher_path: Path,
    inherited_environment: Mapping[str, str],
    ack_timeout: float = START_ACK_TIMEOUT_SECONDS,
) -> RuntimeState:
    read_descriptor, write_descriptor = os.pipe()
    request: ServiceRequest | None = None
    reserved = False
    supervisor: subprocess.Popen[bytes] | None = None
    try:
        _reconcile_before_start(layout, candidate.service_id)
        request = _request(candidate)
        reserve_request(layout, request)
        reserved = True
        supervisor = subprocess.Popen(
            [
                str(launcher_path),
                "supervise",
                "--state-root",
                str(layout.state_root),
                "--target-root",
                str(layout.targets_root),
                "--service-id",
                request.service_id,
                "--request-id",
                request.request_id,
                "--ack-fd",
                str(write_descriptor),
            ],
            stdin=subprocess.DEVNULL,
            stdout=None,
            stderr=None,
            env=dict(inherited_environment),
            pass_fds=(write_descriptor,),
            start_new_session=True,
        )
        os.close(write_descriptor)
        write_descriptor = -1
        acknowledgement = _read_ack(read_descriptor, ack_timeout)
        if acknowledgement.get("ok") is not True:
            detail = acknowledgement.get("error")
            if not isinstance(detail, str) or not detail:
                detail = "supervisor rejected startup"
            raise ServiceError(detail)
        runtime = read_runtime(layout, request.service_id)
        if runtime is None or runtime.request_id != request.request_id:
            raise ServiceError("supervisor acknowledged startup without runtime state")
        return runtime
    except BaseException:
        if reserved and request is not None:
            try:
                request_cancellation(
                    layout,
                    request.service_id,
                    request.request_id,
                )
            except StateError:
                pass
        if supervisor is not None and supervisor.poll() is None:
            supervisor.terminate()
            try:
                supervisor.wait(timeout=START_CANCEL_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                supervisor.kill()
                supervisor.wait(timeout=START_CANCEL_TIMEOUT_SECONDS)
        if reserved and request is not None:
            cleanup_service(
                layout,
                request.service_id,
                expected_request_id=request.request_id,
            )
        raise
    finally:
        os.close(read_descriptor)
        if write_descriptor >= 0:
            os.close(write_descriptor)


def _server_command(
    candidate: CandidateRequest,
    runtime_environment: dict[str, str],
    listen_host: str,
    port: int,
) -> list[str]:
    runtime_environment.update(
        {
            "VLLM_HOST": listen_host,
            "VLLM_PORT": str(port),
            "VLLM_SERVICE_ID": candidate.service_id,
            "VLLM_ROLE": candidate.role,
            "VLLM_MODEL_ALIAS": candidate.model_alias,
            "VLLM_LAUNCH_SCRIPT": "vllm-service-launch",
        }
    )
    if candidate.runtime == "vllm":
        return [
            str(candidate.environment.vllm_executable),
            *candidate.vllm_argv,
            "--host",
            listen_host,
            "--port",
            str(port),
        ]
    runtime_environment["DS_LLM_PROMETHEUS_HOST"] = listen_host
    runtime_environment["DS_LLM_PROMETHEUS_PORT"] = str(port)
    return list(candidate.vllm_argv)


def _identity(pid: int) -> ProcessIdentity:
    observed = read_process(pid)
    if observed is None:
        raise ServiceError(f"process {pid} exited before identity capture")
    return observed[0]


def _signal_identities(
    identities: tuple[ProcessIdentity, ...],
    signal_number: int,
) -> None:
    for identity in identities:
        if not identity_is_alive(identity):
            continue
        try:
            os.kill(identity.pid, signal_number)
        except ProcessLookupError:
            pass


def _wait_identities(
    identities: tuple[ProcessIdentity, ...],
    timeout: float,
) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not any(identity_is_alive(identity) for identity in identities):
            return True
        time.sleep(0.05)
    return not any(identity_is_alive(identity) for identity in identities)


def supervise_service(
    layout: RuntimeLayout,
    service_id: str,
    request_id: str,
    ack_descriptor: int,
    *,
    inherited_environment: Mapping[str, str],
    stop_timeout: float = STOP_TIMEOUT_SECONDS,
) -> int:
    pending_signal: int | None = None
    acknowledgement_sent = False
    server: subprocess.Popen[bytes] | None = None
    runtime: RuntimeState | None = None
    output_file: object | None = None
    claimed = False

    def capture_signal(signal_number: int, _frame: object) -> None:
        nonlocal pending_signal
        pending_signal = signal_number

    signal.signal(signal.SIGTERM, capture_signal)
    signal.signal(signal.SIGINT, capture_signal)

    try:
        request = attach_supervisor(
            layout,
            service_id,
            request_id,
            _identity(os.getpid()),
        )
        claimed = True
        environment = build_runtime_environment(
            request.candidate,
            inherited_environment,
            cancellation_signal=lambda: pending_signal,
        )
        if pending_signal is not None:
            raise ServiceError("startup was cancelled before server spawn")
        with locked_registry(layout):
            endpoint = allocate_endpoint(
                request.candidate.listen_host,
                request.candidate.port_policy,
                reserved_ports=reserved_ports(layout),
            )
            command = _server_command(
                request.candidate,
                environment,
                endpoint.listen_host,
                endpoint.port,
            )
            if pending_signal is not None:
                raise ServiceError("startup was cancelled before server spawn")
            stdout: object | None = None
            stderr: object | int | None = None
            if request.candidate.output is not None:
                output_file = request.candidate.output.open("ab", buffering=0)
                os.fchmod(output_file.fileno(), 0o600)
                stdout = output_file
                stderr = subprocess.STDOUT
            server = subprocess.Popen(
                command,
                cwd=request.candidate.working_directory,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            server_identity = _identity(server.pid)
            runtime = RuntimeState(
                request_id=request.request_id,
                service_id=request.service_id,
                supervisor=_identity(os.getpid()),
                server=server_identity,
                server_pgid=os.getpgid(server.pid),
                server_session_id=os.getsid(server.pid),
                listen_host=endpoint.listen_host,
                scrape_host=endpoint.scrape_host,
                port=endpoint.port,
            )
            publish_runtime(layout, request, runtime)
        _write_ack(ack_descriptor, {"ok": True})
        acknowledgement_sent = True
        os.close(ack_descriptor)
        ack_descriptor = -1
        _log(service_id, f"server_pid={server.pid} state=started")

        forwarded_at: float | None = None
        force_kill_sent = False
        while server.poll() is None:
            if pending_signal is not None and forwarded_at is None:
                signal_process_group(
                    runtime.server,
                    runtime.server_pgid,
                    runtime.server_session_id,
                    pending_signal,
                    anchor=runtime.supervisor,
                )
                forwarded_at = time.monotonic()
            if (
                forwarded_at is not None
                and time.monotonic() - forwarded_at >= stop_timeout
                and not force_kill_sent
            ):
                signal_process_group(
                    runtime.server,
                    runtime.server_pgid,
                    runtime.server_session_id,
                    signal.SIGKILL,
                    anchor=runtime.supervisor,
                )
                force_kill_sent = True
            time.sleep(0.05)
        exit_code = server.wait()
        if process_group_is_alive(
            runtime.server,
            runtime.server_pgid,
            runtime.server_session_id,
            anchor=runtime.supervisor,
        ):
            terminate_process_group(
                runtime.server,
                runtime.server_pgid,
                runtime.server_session_id,
                timeout=stop_timeout,
                anchor=runtime.supervisor,
            )
        _log(service_id, f"server_pid={server.pid} state=exited code={exit_code}")
        return 0 if exit_code == 0 else 1
    except BaseException as exc:
        if runtime is not None:
            terminate_process_group(
                runtime.server,
                runtime.server_pgid,
                runtime.server_session_id,
                timeout=stop_timeout,
                anchor=runtime.supervisor,
            )
        elif server is not None and server.poll() is None:
            server.kill()
            server.wait()
        if not acknowledgement_sent:
            try:
                _write_ack(ack_descriptor, {"ok": False, "error": str(exc)})
            except OSError:
                pass
        print(f"vllm-service-launch[{service_id}]: {exc}", file=sys.stderr)
        return 1
    finally:
        if ack_descriptor >= 0:
            os.close(ack_descriptor)
        if claimed:
            try:
                cleanup_service(
                    layout,
                    service_id,
                    expected_request_id=request_id,
                )
            except Exception as exc:
                print(
                    f"vllm-service-launch[{service_id}]: cleanup failed: {exc}",
                    file=sys.stderr,
                )
        if output_file is not None:
            output_file.close()


def service_status(layout: RuntimeLayout, service_id: str) -> dict[str, object]:
    request = read_request(layout, service_id)
    runtime = read_runtime(layout, service_id)
    if runtime is None:
        pending_owner_alive = identity_is_alive(request.starter) or (
            request.supervisor is not None
            and identity_is_alive(request.supervisor)
        )
        if not pending_owner_alive:
            if not request.cancellation_requested:
                raise ServiceError(
                    "service startup owner exited before supervisor claim; "
                    "run stop to cancel the retained reservation"
                )
            cleanup_service(
                layout,
                service_id,
                expected_request_id=request.request_id,
            )
            raise ServiceError("service startup is no longer running")
        return {
            "service_id": service_id,
            "request_id": request.request_id,
            "state": "starting",
            "runtime": request.candidate.runtime,
        }
    supervisor_alive = identity_is_alive(runtime.supervisor)
    server_alive = _runtime_group_is_alive(runtime)
    if not server_alive:
        _raise_if_ambiguous_orphan(runtime)
        cleanup_service(
            layout,
            service_id,
            expected_request_id=request.request_id,
        )
        raise ServiceError("service is not running")
    state = "running" if supervisor_alive else "orphaned"
    return {
        "service_id": service_id,
        "request_id": request.request_id,
        "state": state,
        "runtime": request.candidate.runtime,
        "supervisor_pid": runtime.supervisor.pid,
        "pid": runtime.server.pid,
        "listen_host": runtime.listen_host,
        "scrape_host": runtime.scrape_host,
        "port": runtime.port,
    }


def stop_service(
    layout: RuntimeLayout,
    service_id: str,
    *,
    expected_request_id: str | None = None,
    stop_timeout: float = STOP_TIMEOUT_SECONDS,
) -> None:
    request = read_request(layout, service_id)
    if expected_request_id is not None and request.request_id != expected_request_id:
        raise ServiceError("request ownership changed before stop")
    runtime = read_runtime(layout, service_id)
    if runtime is None:
        request = request_cancellation(
            layout,
            service_id,
            request.request_id,
        )
        owners = tuple(
            owner
            for owner in (request.supervisor, request.starter)
            if owner is not None and identity_is_alive(owner)
        )
        _signal_identities(owners, signal.SIGTERM)
        if not _wait_identities(owners, stop_timeout):
            _signal_identities(owners, signal.SIGKILL)
        if not _wait_identities(owners, START_CANCEL_TIMEOUT_SECONDS):
            raise ServiceError("startup owners survived SIGKILL")
        cleanup_service(
            layout,
            service_id,
            expected_request_id=request.request_id,
        )
        return

    if identity_is_alive(runtime.supervisor):
        try:
            os.kill(runtime.supervisor.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + stop_timeout
        while time.monotonic() < deadline:
            if not layout.request_path(service_id).exists():
                return
            if not identity_is_alive(runtime.supervisor):
                break
            time.sleep(0.05)

        if identity_is_alive(runtime.supervisor):
            terminate_process_group(
                runtime.server,
                runtime.server_pgid,
                runtime.server_session_id,
                timeout=0,
                anchor=runtime.supervisor,
            )
            supervisor_identity = (runtime.supervisor,)
            _signal_identities(supervisor_identity, signal.SIGKILL)
            if not _wait_identities(
                supervisor_identity,
                START_CANCEL_TIMEOUT_SECONDS,
            ):
                raise ServiceError("supervisor survived SIGKILL")

    _raise_if_ambiguous_orphan(runtime)
    terminate_process_group(
        runtime.server,
        runtime.server_pgid,
        runtime.server_session_id,
        timeout=stop_timeout,
    )
    cleanup_service(
        layout,
        service_id,
        expected_request_id=request.request_id,
    )
