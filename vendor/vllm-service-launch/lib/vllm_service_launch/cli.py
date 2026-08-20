"""Command-line interface for per-service vLLM supervisors."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Sequence

from .endpoint import EndpointError
from .environment import (
    RuntimeEnvironmentError,
    parse_env_files,
    resolve_conda_environment,
    resolve_uv_environment,
)
from .process import ProcessError
from .schema import (
    OPAQUE_ID_PATTERN,
    SERVICE_ID_PATTERN,
    AutoPortPolicy,
    CandidateRequest,
    ContractError,
    FixedPortPolicy,
)
from .service import (
    ServiceError,
    service_status,
    start_service,
    stop_service,
    supervise_service,
)
from .state import RuntimeLayout, StateError


class CliError(RuntimeError):
    """A CLI operation cannot be completed."""


ROOT_DESCRIPTION = """\
启动和管理无需 systemd 的一次性 vLLM 或 DashLLM 服务。

每个服务由一个独立 launcher supervisor 监控。服务默认继承调用方完整环境和
stdout/stderr，结束后 supervisor 清理 lifecycle metadata 与 Prometheus target 并退出。"""

START_EPILOG = """\
端口规则：
  不指定 --port 或 --port-range 时，本次运行从 8000:8099 自动选择可用端口。
  --port 使用固定端口；--port-range 使用 START:END 范围。

示例：
  vllm-service-launch start --service-id qwen35-prefill --role prefill \\
    --model-alias Qwen3.5-397B-A17B-FP8 --uv-project /work/vllm \\
    -- serve /models/Qwen3.5-397B-A17B-FP8

注意：
  -- 后的参数会原样传给 runtime；vLLM 模式不要重复指定 --host 或 --port。
  DashLLM 模式要求首个参数是绝对 executable。"""


def _service_id(value: str) -> str:
    if SERVICE_ID_PATTERN.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("invalid service ID")
    return value


def _request_id(value: str) -> str:
    if OPAQUE_ID_PATTERN.fullmatch(value) is None:
        raise argparse.ArgumentTypeError(
            "request ID must be 32 lowercase hex characters"
        )
    return value


def _positive_fd(value: str) -> int:
    try:
        descriptor = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ack fd must be an integer") from exc
    if descriptor < 0:
        raise argparse.ArgumentTypeError("ack fd must be non-negative")
    return descriptor


def _port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port must be an integer") from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be in 1:65535")
    return port


def _port_range(value: str) -> tuple[int, int]:
    parts = value.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("port range must be START:END")
    start = _port(parts[0])
    end = _port(parts[1])
    if start > end:
        raise argparse.ArgumentTypeError("port range start exceeds end")
    return start, end


def _add_service_id(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--service-id",
        required=True,
        type=_service_id,
        help="服务唯一标识，只允许小写字母、数字和连字符",
    )


def _add_layout(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--state-root",
        type=Path,
        default=Path.cwd() / ".state" / "vllm-service-launch",
        help="私有 lifecycle state 根目录",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=Path("/run/vllm-metrics-targets/targets"),
        help="Prometheus file-SD target 目录",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vllm-service-launch",
        description=ROOT_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        title="commands",
        metavar="{start,status,stop}",
    )

    start = subparsers.add_parser(
        "start",
        help="启动一次新的 runtime 服务",
        epilog=START_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_layout(start)
    _add_service_id(start)
    start.add_argument("--runtime", choices=("vllm", "dashllm"), default="vllm")
    start.add_argument("--output", type=Path, metavar="ABSOLUTE_PATH")
    start.add_argument("--role", required=True, choices=("prefill", "decode", "mixed"))
    start.add_argument("--model-alias", required=True)
    environment = start.add_mutually_exclusive_group(required=True)
    environment.add_argument("--conda-env")
    environment.add_argument("--uv-project", type=Path)
    start.add_argument("--env-file", action="append", type=Path, default=[])
    start.add_argument("--working-directory", type=Path)
    start.add_argument("--host", default="127.0.0.1")
    ports = start.add_mutually_exclusive_group()
    ports.add_argument("--port", type=_port)
    ports.add_argument("--port-range", type=_port_range, metavar="START:END")
    start.add_argument("runtime_argv", nargs=argparse.REMAINDER, metavar="RUNTIME_ARG")

    status = subparsers.add_parser("status", help="查看服务状态和实际监听端口")
    _add_layout(status)
    _add_service_id(status)
    status.add_argument("--json", action="store_true")

    stop = subparsers.add_parser("stop", help="停止服务并清理本次运行状态")
    _add_layout(stop)
    _add_service_id(stop)
    stop.add_argument("--request-id", type=_request_id)

    supervise = subparsers.add_parser("supervise")
    _add_layout(supervise)
    _add_service_id(supervise)
    supervise.add_argument("--request-id", required=True, type=_request_id)
    supervise.add_argument("--ack-fd", required=True, type=_positive_fd)
    return parser


def _resolved_existing_file(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise CliError(f"file does not exist: {resolved}")
    return resolved


def _resolved_existing_directory(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_dir():
        raise CliError(f"directory does not exist: {resolved}")
    return resolved


def _candidate_from_args(args: argparse.Namespace) -> CandidateRequest:
    if args.conda_env is not None:
        environment = resolve_conda_environment(args.conda_env)
    else:
        environment = resolve_uv_environment(
            _resolved_existing_directory(args.uv_project)
        )
    env_files = tuple(_resolved_existing_file(path) for path in args.env_file)
    parse_env_files(env_files, {})
    working_directory = _resolved_existing_directory(
        args.working_directory if args.working_directory is not None else Path.cwd()
    )
    if args.port is not None:
        policy = FixedPortPolicy(mode="fixed", port=args.port)
    else:
        start, end = args.port_range if args.port_range is not None else (8000, 8099)
        policy = AutoPortPolicy(mode="auto", range_start=start, range_end=end)
    if not args.runtime_argv or args.runtime_argv[0] != "--":
        raise CliError("runtime argv must follow --")
    runtime_argv = args.runtime_argv[1:]
    if not runtime_argv:
        raise CliError("runtime argv must follow --")
    payload: dict[str, object] = {
        "schema_version": 1,
        "service_id": args.service_id,
        "role": args.role,
        "model_alias": args.model_alias,
        "environment": environment.to_dict(),
        "env_files": [str(path) for path in env_files],
        "working_directory": str(working_directory),
        "listen_host": args.host,
        "port_policy": policy.to_dict(),
        "runtime": args.runtime,
        "vllm_argv": runtime_argv,
    }
    if args.output is not None:
        payload["output"] = str(args.output)
    return CandidateRequest.from_dict(payload)


def _print_status(payload: dict[str, object], *, as_json: bool) -> None:
    if as_json:
        json.dump(payload, sys.stdout, ensure_ascii=False, sort_keys=True)
        sys.stdout.write("\n")
        return
    endpoint = ""
    if "port" in payload:
        endpoint = f" endpoint={payload['scrape_host']}:{payload['port']}"
    print(f"{payload['service_id']} state={payload['state']}{endpoint}")


def _absolute_without_resolving_symlinks(path: Path) -> Path:
    return Path(os.path.abspath(path))


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(argv) if argv is not None else sys.argv[1:]
    args = build_parser().parse_args(arguments)
    layout = RuntimeLayout.from_roots(
        _absolute_without_resolving_symlinks(args.state_root),
        _absolute_without_resolving_symlinks(args.target_root),
    )
    try:
        if args.command == "start":
            candidate = _candidate_from_args(args)
            runtime = start_service(
                layout,
                candidate,
                launcher_path=Path(sys.argv[0]).resolve(),
                inherited_environment=os.environ,
            )
            _print_status(
                {
                    "service_id": runtime.service_id,
                    "request_id": runtime.request_id,
                    "state": "started",
                    "scrape_host": runtime.scrape_host,
                    "port": runtime.port,
                },
                as_json=False,
            )
            return 0
        if args.command == "status":
            _print_status(
                service_status(layout, args.service_id),
                as_json=args.json,
            )
            return 0
        if args.command == "stop":
            stop_service(
                layout,
                args.service_id,
                expected_request_id=args.request_id,
            )
            return 0
        if args.command == "supervise":
            return supervise_service(
                layout,
                args.service_id,
                args.request_id,
                args.ack_fd,
                inherited_environment=os.environ,
            )
        raise CliError("unsupported command")
    except (
        CliError,
        ContractError,
        EndpointError,
        RuntimeEnvironmentError,
        ProcessError,
        ServiceError,
        StateError,
        OSError,
    ) as exc:
        print(f"vllm-service-launch: {exc}", file=sys.stderr)
        return 1
