"""Private lifecycle metadata and public Prometheus target state."""

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

from .endpoint import Endpoint
from .schema import ContractError, ProcessIdentity, RuntimeState, ServiceRequest


class StateError(RuntimeError):
    """Launcher lifecycle state is missing, malformed, or unsafe."""


@dataclass(frozen=True)
class RuntimeLayout:
    state_root: Path
    targets_root: Path

    @classmethod
    def from_roots(
        cls,
        state_root: Path,
        targets_root: Path,
    ) -> RuntimeLayout:
        return cls(state_root=state_root, targets_root=targets_root)

    @property
    def services_root(self) -> Path:
        return self.state_root / "services"

    @property
    def registry_lock(self) -> Path:
        return self.state_root / "registry.lock"

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


def _ensure_private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise StateError(f"private state path is not a real directory: {path}")
    os.chmod(path, 0o700)


def _prepare_service(layout: RuntimeLayout, service_id: str) -> None:
    _ensure_private_directory(layout.state_root)
    _ensure_private_directory(layout.services_root)
    _ensure_private_directory(layout.service_directory(service_id))


def _require_target_root(layout: RuntimeLayout) -> None:
    try:
        metadata = layout.targets_root.lstat()
    except FileNotFoundError as exc:
        raise StateError(f"target root does not exist: {layout.targets_root}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise StateError(f"target root is not a real directory: {layout.targets_root}")
    if not os.access(layout.targets_root, os.W_OK):
        raise StateError(f"target root is not writable: {layout.targets_root}")


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(
        directory,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
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
            os.fchmod(temporary.fileno(), mode)
            json.dump(payload, temporary, ensure_ascii=False, indent=2)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
        _fsync_directory(path.parent)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _read_json(path: Path) -> object:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
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


def _unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    _fsync_directory(path.parent)


@contextmanager
def _exclusive_lock(path: Path, *, create: bool) -> Iterator[None]:
    flags = os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW
    if create:
        flags |= os.O_CREAT
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


@contextmanager
def locked_service(
    layout: RuntimeLayout,
    service_id: str,
    *,
    create: bool = True,
) -> Iterator[None]:
    if create:
        _prepare_service(layout, service_id)
    with _exclusive_lock(layout.state_lock(service_id), create=create):
        yield


@contextmanager
def locked_registry(layout: RuntimeLayout) -> Iterator[None]:
    _ensure_private_directory(layout.state_root)
    _ensure_private_directory(layout.services_root)
    with _exclusive_lock(layout.registry_lock, create=True):
        yield


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


def read_request(layout: RuntimeLayout, service_id: str) -> ServiceRequest:
    return _read_request(layout, service_id)


def read_runtime(
    layout: RuntimeLayout,
    service_id: str,
) -> RuntimeState | None:
    try:
        layout.runtime_path(service_id).lstat()
    except FileNotFoundError:
        return None
    return _read_runtime(layout, service_id)


def reserve_request(layout: RuntimeLayout, request: ServiceRequest) -> None:
    with locked_service(layout, request.service_id):
        try:
            layout.request_path(request.service_id).lstat()
        except FileNotFoundError:
            pass
        else:
            raise StateError(f"service is already reserved: {request.service_id}")
        _atomic_write_json(
            layout.request_path(request.service_id),
            request.to_dict(),
            0o600,
        )


def attach_supervisor(
    layout: RuntimeLayout,
    service_id: str,
    request_id: str,
    supervisor: ProcessIdentity,
) -> ServiceRequest:
    with locked_service(layout, service_id, create=False):
        request = _read_request(layout, service_id)
        if request.request_id != request_id:
            raise StateError("request ownership changed before supervisor attachment")
        if request.supervisor is not None:
            raise StateError("request already has a supervisor")
        if request.cancellation_requested:
            raise StateError("request was cancelled before supervisor attachment")
        updated = ServiceRequest(
            request_id=request.request_id,
            candidate=request.candidate,
            starter=request.starter,
            supervisor=supervisor,
            cancellation_requested=False,
        )
        _atomic_write_json(
            layout.request_path(service_id),
            updated.to_dict(),
            0o600,
        )
        return updated


def request_cancellation(
    layout: RuntimeLayout,
    service_id: str,
    request_id: str,
) -> ServiceRequest:
    """Atomically prevent an unclaimed startup from publishing runtime state."""

    with locked_service(layout, service_id, create=False):
        request = _read_request(layout, service_id)
        if request.request_id != request_id:
            raise StateError("request ownership changed before cancellation")
        if request.cancellation_requested:
            return request
        updated = ServiceRequest(
            request_id=request.request_id,
            candidate=request.candidate,
            starter=request.starter,
            supervisor=request.supervisor,
            cancellation_requested=True,
        )
        _atomic_write_json(
            layout.request_path(service_id),
            updated.to_dict(),
            0o600,
        )
        return updated


def reserved_ports(layout: RuntimeLayout) -> frozenset[int]:
    ports: set[int] = set()
    if not layout.services_root.exists():
        return frozenset()
    for service_directory in layout.services_root.iterdir():
        if service_directory.is_symlink() or not service_directory.is_dir():
            continue
        runtime_path = service_directory / "runtime.json"
        try:
            runtime_path.lstat()
        except FileNotFoundError:
            continue
        try:
            runtime = RuntimeState.from_dict(_read_json(runtime_path))
        except ContractError as exc:
            raise StateError(
                f"runtime file violates its schema: {runtime_path}"
            ) from exc
        ports.add(runtime.port)
    return frozenset(ports)


def publish_runtime(
    layout: RuntimeLayout,
    request: ServiceRequest,
    runtime: RuntimeState,
) -> None:
    if runtime.request_id != request.request_id:
        raise StateError("runtime request ID does not match request")
    if runtime.service_id != request.service_id:
        raise StateError("runtime service ID does not match request")
    _require_target_root(layout)
    with locked_service(layout, request.service_id, create=False):
        current = _read_request(layout, request.service_id)
        if current != request:
            raise StateError("request changed before runtime publication")
        if (
            request.cancellation_requested
            or request.supervisor is None
            or request.supervisor != runtime.supervisor
        ):
            raise StateError("runtime publication requires the claimed supervisor")
        endpoint = Endpoint(
            listen_host=runtime.listen_host,
            scrape_host=runtime.scrape_host,
            port=runtime.port,
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
        try:
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
        except BaseException:
            _unlink(layout.target_path(request.service_id))
            _unlink(layout.runtime_path(request.service_id))
            raise


def cleanup_service(
    layout: RuntimeLayout,
    service_id: str,
    *,
    expected_request_id: str,
) -> bool:
    try:
        layout.service_directory(service_id).lstat()
    except FileNotFoundError:
        return False
    with locked_registry(layout):
        with locked_service(layout, service_id, create=False):
            try:
                request = _read_request(layout, service_id)
            except StateError:
                if layout.request_path(service_id).exists():
                    raise
                return False
            if request.request_id != expected_request_id:
                return False
            runtime = read_runtime(layout, service_id)
            if runtime is not None and runtime.request_id != request.request_id:
                raise StateError("runtime ownership does not match request")
            _unlink(layout.target_path(service_id))
            _unlink(layout.runtime_path(service_id))
            _unlink(layout.request_path(service_id))
            return True
