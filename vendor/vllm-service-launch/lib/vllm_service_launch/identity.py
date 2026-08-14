"""Trusted sudo caller identity and privilege dropping."""

from __future__ import annotations

import os
import pwd
from typing import Callable, Mapping

from .schema import CallerIdentity, ContractError


class IdentityError(RuntimeError):
    """The launcher cannot establish a trusted non-root caller."""


def _sudo_integer(
    environment: Mapping[str, str],
    name: str,
) -> int:
    value = environment.get(name)
    if value is None or not value.isdecimal():
        raise IdentityError(f"{name} is required")
    return int(value)


def caller_from_sudo_environment(
    environment: Mapping[str, str],
    *,
    effective_uid: int,
) -> CallerIdentity:
    if effective_uid != 0:
        raise IdentityError("internal command requires root")
    uid = _sudo_integer(environment, "SUDO_UID")
    gid = _sudo_integer(environment, "SUDO_GID")
    if uid == 0:
        raise IdentityError("start requires a non-root caller")
    try:
        record = pwd.getpwuid(uid)
    except KeyError as exc:
        raise IdentityError("sudo caller does not exist") from exc
    try:
        return CallerIdentity.from_dict(
            {
                "uid": uid,
                "gid": gid,
                "name": record.pw_name,
                "home": record.pw_dir,
                "shell": record.pw_shell,
            }
        )
    except ContractError as exc:
        raise IdentityError("sudo caller account is invalid") from exc


def drop_privileges(
    caller: CallerIdentity,
    *,
    initgroups: Callable[[str, int], None] = os.initgroups,
    setgid: Callable[[int], None] = os.setgid,
    setuid: Callable[[int], None] = os.setuid,
) -> None:
    initgroups(caller.name, caller.gid)
    setgid(caller.gid)
    setuid(caller.uid)
