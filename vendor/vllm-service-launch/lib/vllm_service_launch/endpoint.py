"""Listen and scrape endpoint allocation."""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from typing import AbstractSet

from .schema import AutoPortPolicy, FixedPortPolicy, PortPolicy


class EndpointError(RuntimeError):
    """No valid endpoint can be allocated."""


@dataclass(frozen=True)
class Endpoint:
    listen_host: str
    scrape_host: str
    port: int

    @property
    def prometheus_target(self) -> str:
        address = ipaddress.ip_address(self.scrape_host)
        if address.version == 6:
            return f"[{self.scrape_host}]:{self.port}"
        return f"{self.scrape_host}:{self.port}"


def derive_scrape_host(listen_host: str) -> str:
    try:
        address = ipaddress.ip_address(listen_host)
    except ValueError as exc:
        raise EndpointError("listen_host must be an IP literal") from exc
    if address.is_unspecified:
        return "127.0.0.1" if address.version == 4 else "::1"
    return listen_host


def _socket_address(listen_host: str, port: int) -> tuple[object, ...]:
    address = ipaddress.ip_address(listen_host)
    if address.version == 6:
        return (listen_host, port, 0, 0)
    return (listen_host, port)


def _can_bind(listen_host: str, port: int) -> bool:
    address = ipaddress.ip_address(listen_host)
    family = socket.AF_INET6 if address.version == 6 else socket.AF_INET
    try:
        with socket.socket(family, socket.SOCK_STREAM) as probe:
            probe.bind(_socket_address(listen_host, port))
    except OSError:
        return False
    return True


def allocate_endpoint(
    listen_host: str,
    policy: PortPolicy,
    *,
    reserved_ports: AbstractSet[int],
) -> Endpoint:
    scrape_host = derive_scrape_host(listen_host)
    if isinstance(policy, FixedPortPolicy):
        candidates = (policy.port,)
    elif isinstance(policy, AutoPortPolicy):
        candidates = range(policy.range_start, policy.range_end + 1)
    else:
        raise TypeError("unsupported port policy")

    for port in candidates:
        if port in reserved_ports:
            continue
        if _can_bind(listen_host, port):
            return Endpoint(
                listen_host=listen_host,
                scrape_host=scrape_host,
                port=port,
            )

    if isinstance(policy, FixedPortPolicy):
        raise EndpointError(f"port {policy.port} is not available")
    raise EndpointError(f"no available port in {policy.range_start}:{policy.range_end}")
