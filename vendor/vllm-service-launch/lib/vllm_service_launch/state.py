"""Ephemeral request, runtime, and Prometheus target state."""

from __future__ import annotations

import fcntl
import json
import os
import stat
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .endpoint import allocate_endpoint
from .schema import (
    CallerIdentity,
    CandidateRequest,
    ContractError,
    RuntimeState,
    ServiceRequest,
)


class StateError(RuntimeError):
    """Ephemeral launcher state is missing, malformed, or not owned."""


@dataclass(frozen=True)
class RuntimeLayout:
    services_root: Path
    targets_root: Path

    @classmethod
    def system(cls) -> RuntimeLayout:
        return cls(
            services_root=Path("/run/vllm-services"),
            targets_root=Path("/run/vllm-metrics-targets/targets"),
        )

    @classmethod
    def under(cls, root: Path) -> RuntimeLayout:
        return cls(
            services_root=root / "vllm-services",
            targets_root=root / "vllm-metrics-targets" / "targets",
        )

    @property
    def registry_lock(self) -> Path:
        return self.services_root / "registry.lock"

    def service_directory(self, service_id: str) -> Path:
        return self.services_root / service_id

    def state_lock(self, service_id: str) -> Path:
        return self.service_directory(service_id) / "state.lock"

    def request_path(self, service_id: str) -> Path:
        return self.service_directory(service_id) / "request.json"

    def runtime_path(self, service_id: str) -> Path:
        return self.service_directory(service_id) / "runtime.json"

    def target_path(self, service_id: str) -> Path:
        return self.targets_root / f"{service_id}.json"


def _ensure_directory(path: Path, mode: int) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=mode)
    os.chmod(path, mode)


def _prepare_layout(layout: RuntimeLayout) -> None:
    _ensure_directory(layout.services_root, 0o755)
    _ensure_directory(layout.targets_root.parent, 0o755)
    _ensure_directory(layout.targets_root, 0o755)


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(
        directory,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_json(path: Path, payload: Any, mode: int) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(payload, temporary, ensure_ascii=False, indent=2)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.chmod(temporary_path, mode)
        os.replace(temporary_path, path)
        temporary_path = None
        _fsync_directory(path.parent)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _read_json(path: Path) -> object:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise StateError(f"cannot open state file: {path}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise StateError(f"state path is not a regular file: {path}")
        with os.fdopen(descriptor, encoding="utf-8") as state_file:
            descriptor = -1
            try:
                return json.load(state_file)
            except json.JSONDecodeError as exc:
                raise StateError(f"state file contains invalid JSON: {path}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _read_request(layout: RuntimeLayout, service_id: str) -> ServiceRequest:
    try:
        return ServiceRequest.from_dict(_read_json(layout.request_path(service_id)))
    except ContractError as exc:
        raise StateError("request file violates its schema") from exc


def _read_runtime(layout: RuntimeLayout, service_id: str) -> RuntimeState:
    try:
        return RuntimeState.from_dict(_read_json(layout.runtime_path(service_id)))
    except ContractError as exc:
        raise StateError("runtime file violates its schema") from exc


@contextmanager
def _exclusive_lock(path: Path, *, create: bool) -> Iterator[None]:
    flags = os.O_RDWR | os.O_CLOEXEC
    if create:
        flags |= os.O_CREAT
    flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise StateError(f"cannot open lock: {path}") from exc
    try:
        os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def reserve_request(
    layout: RuntimeLayout,
    candidate: CandidateRequest,
    caller: CallerIdentity,
    *,
    request_id: str,
) -> ServiceRequest:
    request = ServiceRequest.pending(candidate, caller, request_id)
    _prepare_layout(layout)
    service_directory = layout.service_directory(candidate.service_id)
    try:
        service_directory.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise StateError(
            f"service is already reserved: {candidate.service_id}"
        ) from exc
    os.chmod(service_directory, 0o700)
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
        flags |= os.O_NOFOLLOW
        lock_descriptor = os.open(
            layout.state_lock(candidate.service_id),
            flags,
            0o600,
        )
        try:
            os.fchmod(lock_descriptor, 0o600)
        finally:
            os.close(lock_descriptor)
        _atomic_write_json(
            layout.request_path(candidate.service_id),
            request.to_dict(),
            0o600,
        )
    except BaseException:
        layout.request_path(candidate.service_id).unlink(missing_ok=True)
        layout.state_lock(candidate.service_id).unlink(missing_ok=True)
        service_directory.rmdir()
        raise
    return request


def claim_request(
    layout: RuntimeLayout,
    service_id: str,
    *,
    invocation_id: str,
) -> ServiceRequest:
    with _exclusive_lock(layout.state_lock(service_id), create=False):
        request = _read_request(layout, service_id)
        claimed = request.claimed(invocation_id)
        _atomic_write_json(
            layout.request_path(service_id),
            claimed.to_dict(),
            0o600,
        )
        return claimed


def discard_pending(
    layout: RuntimeLayout,
    service_id: str,
    *,
    request_id: str,
) -> bool:
    service_directory = layout.service_directory(service_id)
    if not service_directory.exists():
        return False
    with _exclusive_lock(layout.state_lock(service_id), create=False):
        request = _read_request(layout, service_id)
        if request.state != "pending" or request.request_id != request_id:
            return False
        _unlink(layout.request_path(service_id))
    _unlink(layout.state_lock(service_id))
    try:
        service_directory.rmdir()
    except OSError as exc:
        raise StateError(
            f"service directory is not empty: {service_directory}"
        ) from exc
    _fsync_directory(layout.services_root)
    return True


def _reserved_ports(layout: RuntimeLayout) -> frozenset[int]:
    ports: set[int] = set()
    for service_directory in layout.services_root.iterdir():
        if not service_directory.is_dir():
            continue
        runtime_path = service_directory / "runtime.json"
        if not runtime_path.exists():
            continue
        try:
            runtime = RuntimeState.from_dict(_read_json(runtime_path))
        except ContractError as exc:
            raise StateError(
                f"runtime file violates its schema: {runtime_path}"
            ) from exc
        ports.add(runtime.port)
    return frozenset(ports)


def allocate_and_publish(
    layout: RuntimeLayout,
    request: ServiceRequest,
    *,
    pid: int,
) -> RuntimeState:
    if request.state != "claimed" or request.invocation_id is None:
        raise StateError("request must be claimed before publishing")
    _prepare_layout(layout)
    with _exclusive_lock(layout.registry_lock, create=True):
        with _exclusive_lock(
            layout.state_lock(request.service_id),
            create=False,
        ):
            current = _read_request(layout, request.service_id)
            if current != request:
                raise StateError("request changed before endpoint publication")
            endpoint = allocate_endpoint(
                request.candidate.listen_host,
                request.candidate.port_policy,
                reserved_ports=_reserved_ports(layout),
            )
            runtime = RuntimeState(
                request_id=request.request_id,
                invocation_id=request.invocation_id,
                service_id=request.service_id,
                pid=pid,
                listen_host=endpoint.listen_host,
                scrape_host=endpoint.scrape_host,
                port=endpoint.port,
            )
            target = [
                {
                    "targets": [endpoint.prometheus_target],
                    "labels": {
                        "service_id": request.service_id,
                        "role": request.candidate.role,
                        "model_alias": request.candidate.model_alias,
                    },
                }
            ]
            _atomic_write_json(
                layout.runtime_path(request.service_id),
                runtime.to_dict(),
                0o600,
            )
            _atomic_write_json(
                layout.target_path(request.service_id),
                target,
                0o644,
            )
            _atomic_write_json(
                layout.request_path(request.service_id),
                request.published().to_dict(),
                0o600,
            )
            return runtime


def _unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    _fsync_directory(path.parent)


def cleanup_invocation(
    layout: RuntimeLayout,
    service_id: str,
    *,
    invocation_id: str,
) -> bool:
    service_directory = layout.service_directory(service_id)
    if not service_directory.exists():
        return False
    _prepare_layout(layout)
    with _exclusive_lock(layout.registry_lock, create=True):
        if not service_directory.exists():
            return False
        with _exclusive_lock(layout.state_lock(service_id), create=False):
            request = _read_request(layout, service_id)
            if request.invocation_id != invocation_id:
                return False
            runtime_path = layout.runtime_path(service_id)
            if runtime_path.exists():
                runtime = _read_runtime(layout, service_id)
                if (
                    runtime.request_id != request.request_id
                    or runtime.invocation_id != invocation_id
                    or runtime.service_id != service_id
                ):
                    raise StateError("runtime ownership does not match request")
            _unlink(layout.target_path(service_id))
            _unlink(runtime_path)
            _unlink(layout.request_path(service_id))

    _unlink(layout.state_lock(service_id))
    try:
        service_directory.rmdir()
    except OSError as exc:
        raise StateError(
            f"service directory is not empty: {service_directory}"
        ) from exc
    _fsync_directory(layout.services_root)
    return True


def read_request(
    layout: RuntimeLayout,
    service_id: str,
) -> ServiceRequest:
    return _read_request(layout, service_id)


@contextmanager
def locked_request(
    layout: RuntimeLayout,
    service_id: str,
) -> Iterator[ServiceRequest]:
    with _exclusive_lock(layout.state_lock(service_id), create=False):
        yield _read_request(layout, service_id)


def read_runtime(
    layout: RuntimeLayout,
    service_id: str,
) -> RuntimeState | None:
    path = layout.runtime_path(service_id)
    if not path.exists():
        return None
    return _read_runtime(layout, service_id)
