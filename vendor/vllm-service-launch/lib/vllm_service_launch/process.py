"""Linux process identity parsing for supervised service lifecycles."""

from __future__ import annotations

import os
from pathlib import Path
import signal
import time

from .schema import ProcessIdentity


class ProcessError(RuntimeError):
    """Linux process metadata is malformed or unavailable."""


def parse_proc_stat(line: str) -> tuple[ProcessIdentity, int, int, str]:
    """Parse identity, process group, session and state from proc stat."""

    opening = line.find("(")
    closing = line.rfind(")")
    if opening <= 0 or closing <= opening:
        raise ProcessError("process stat has an invalid command field")
    try:
        pid = int(line[:opening].strip())
    except ValueError as exc:
        raise ProcessError("process stat has an invalid pid") from exc
    fields = line[closing + 1 :].split()
    if len(fields) < 20:
        raise ProcessError("process stat omits required fields")
    try:
        process_group = int(fields[2])
        session_id = int(fields[3])
        start_time = int(fields[19])
    except ValueError as exc:
        raise ProcessError("process stat contains a non-integer field") from exc
    return (
        ProcessIdentity(pid=pid, start_time=start_time),
        process_group,
        session_id,
        fields[0],
    )


def identity_matches(expected: ProcessIdentity, observed: ProcessIdentity) -> bool:
    """Return whether an observation is the exact Linux process instance."""

    return expected == observed


def read_process(
    pid: int,
) -> tuple[ProcessIdentity, int, int, str] | None:
    """Read one process without treating ordinary process exit as an error."""

    try:
        line = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ProcessError(f"cannot read process identity for pid {pid}") from exc
    return parse_proc_stat(line)


def identity_is_alive(expected: ProcessIdentity) -> bool:
    observed = read_process(expected.pid)
    return (
        observed is not None
        and observed[3] != "Z"
        and identity_matches(expected, observed[0])
    )


def group_members(process_group: int, session_id: int) -> tuple[int, ...]:
    members: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        observed = read_process(int(entry.name))
        if observed is None:
            continue
        identity, observed_group, observed_session, state = observed
        if (
            state != "Z"
            and observed_group == process_group
            and observed_session == session_id
        ):
            members.append(identity.pid)
    return tuple(sorted(members))


def process_group_is_alive(
    leader: ProcessIdentity,
    process_group: int,
    session_id: int,
    *,
    anchor: ProcessIdentity | None = None,
) -> bool:
    observed = read_process(leader.pid)
    if observed is None:
        return anchor is not None and identity_is_alive(anchor) and bool(
            group_members(process_group, session_id)
        )
    if (
        not identity_matches(leader, observed[0])
        or observed[1] != process_group
        or observed[2] != session_id
    ):
        return False
    if observed[3] != "Z":
        return True
    return bool(group_members(process_group, session_id))


def unowned_process_group_is_observed(
    leader: ProcessIdentity,
    process_group: int,
    session_id: int,
) -> bool:
    """Return whether numeric group members exist without the recorded leader."""

    observed = read_process(leader.pid)
    if observed is not None and (
        identity_matches(leader, observed[0])
        and observed[1] == process_group
        and observed[2] == session_id
    ):
        return False
    return bool(group_members(process_group, session_id))


def signal_process_group(
    leader: ProcessIdentity,
    process_group: int,
    session_id: int,
    signal_number: int,
    *,
    anchor: ProcessIdentity | None = None,
) -> bool:
    if not process_group_is_alive(
        leader,
        process_group,
        session_id,
        anchor=anchor,
    ):
        return False
    try:
        os.killpg(process_group, signal_number)
    except ProcessLookupError:
        return False
    return True


def terminate_process_group(
    leader: ProcessIdentity,
    process_group: int,
    session_id: int,
    *,
    timeout: float,
    kill_timeout: float = 5.0,
    anchor: ProcessIdentity | None = None,
) -> None:
    signal_process_group(
        leader,
        process_group,
        session_id,
        signal.SIGTERM,
        anchor=anchor,
    )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not process_group_is_alive(
            leader,
            process_group,
            session_id,
            anchor=anchor,
        ):
            return
        time.sleep(0.05)
    signal_process_group(
        leader,
        process_group,
        session_id,
        signal.SIGKILL,
        anchor=anchor,
    )
    kill_deadline = time.monotonic() + kill_timeout
    while time.monotonic() < kill_deadline:
        if not process_group_is_alive(
            leader,
            process_group,
            session_id,
            anchor=anchor,
        ):
            return
        time.sleep(0.01)
    raise ProcessError(f"process group {process_group} survived SIGKILL")
