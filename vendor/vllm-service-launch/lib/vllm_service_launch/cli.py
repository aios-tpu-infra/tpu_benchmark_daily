"""Command-line interface for stateless vLLM services."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Sequence

from .endpoint import EndpointError
from .environment import (
    RuntimeEnvironmentError,
    build_runtime_environment,
    parse_env_files,
    resolve_conda_environment,
    resolve_uv_environment,
)
from .identity import IdentityError, caller_from_sudo_environment, drop_privileges
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
    SystemdSubprocessController,
    redirect_output,
    run_service,
    service_status,
    stop_service,
    submit_request,
)
from .state import RuntimeLayout, StateError, cleanup_invocation


class CliError(RuntimeError):
    """A CLI operation cannot be completed."""


ROOT_DESCRIPTION = """\
启动和管理由 systemd 托管的一次性 vLLM 或 DashLLM 服务。

start 会解析当前用户选择的 Conda 或 uv 环境，把本次启动请求交给 systemd，
并为 Prometheus Agent 发布本次运行的动态采集目标。请求和运行状态只保存在 /run，
服务退出后不会留下供下次启动复用的配置。"""

ROOT_EPILOG = """\
常用操作：
  启动服务并查看完整参数：vllm-service-launch start --help
  查看服务：              vllm-service-launch status --service-id SERVICE_ID
  停止服务：              vllm-service-launch stop --service-id SERVICE_ID"""

START_EPILOG = """\
端口规则：
  不指定 --port 或 --port-range 时，本次运行从 8000:8099 自动选择可用端口。
  --port 使用固定端口；--port-range 使用 START:END 范围。三者只影响本次运行。

示例（Conda）：
  vllm-service-launch start --service-id qwen35-prefill --role prefill \\
    --model-alias Qwen3.5-397B-A17B-FP8 --conda-env torch_tpu \\
    -- serve /models/Qwen3.5-397B-A17B-FP8

示例（uv）：
  vllm-service-launch start --service-id qwen35-decode --role decode \\
    --model-alias Qwen3.5-397B-A17B-FP8 --uv-project /work/vllm \\
    --port-range 8100:8199 -- serve /models/Qwen3.5-397B-A17B-FP8

注意：
  -- 后的参数会原样传给所选 runtime；vLLM 模式不要重复指定 --host 或 --port。
  serve 模式必须保留 HTTP /metrics，不允许 --headless、--grpc 或小于 1 的
  --api-server-count。DashLLM 模式要求首个参数是绝对 executable，采集 endpoint
  通过 DS_LLM_PROMETHEUS_HOST/PORT 注入。"""


def _service_id(value: str) -> str:
    if SERVICE_ID_PATTERN.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("invalid service ID")
    return value


def _port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port must be an integer") from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be in 1:65535")
    return port


def _request_id(value: str) -> str:
    if OPAQUE_ID_PATTERN.fullmatch(value) is None:
        raise argparse.ArgumentTypeError(
            "request ID must be 32 lowercase hex characters"
        )
    return value


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vllm-service-launch",
        description=ROOT_DESCRIPTION,
        epilog=ROOT_EPILOG,
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
        description="启动一次新的无状态 vLLM 或 DashLLM 服务。",
        epilog=START_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_service_id(start)
    start.add_argument(
        "--runtime",
        choices=("vllm", "dashllm"),
        default="vllm",
        help="运行时类型（默认：vllm）",
    )
    start.add_argument(
        "--output",
        type=Path,
        metavar="ABSOLUTE_PATH",
        help="合并 stdout/stderr 的绝对日志路径",
    )
    start.add_argument(
        "--role",
        required=True,
        choices=("prefill", "decode", "mixed"),
        help="监控角色，用于生成 Prometheus job 标签",
    )
    start.add_argument(
        "--model-alias",
        required=True,
        help="写入 Prometheus target 的稳定模型名称",
    )
    environment = start.add_mutually_exclusive_group(required=True)
    environment.add_argument(
        "--conda-env",
        help="Conda 环境名或绝对 prefix",
    )
    environment.add_argument(
        "--uv-project",
        type=Path,
        help="已有 uv 环境的项目目录",
    )
    start.add_argument(
        "--env-file",
        action="append",
        type=Path,
        default=[],
        help="运行环境文件；可重复指定，后面的文件覆盖前面的同名变量",
    )
    start.add_argument(
        "--working-directory",
        type=Path,
        help="runtime 工作目录，默认使用当前目录",
    )
    start.add_argument(
        "--host",
        default="127.0.0.1",
        help="metrics 监听地址（默认：127.0.0.1）",
    )
    ports = start.add_mutually_exclusive_group()
    ports.add_argument(
        "--port",
        type=_port,
        help="本次运行使用的固定端口",
    )
    ports.add_argument(
        "--port-range",
        type=_port_range,
        metavar="START:END",
        help="本次运行的自动端口范围（默认：8000:8099）",
    )
    start.add_argument(
        "vllm_argv",
        nargs=argparse.REMAINDER,
        metavar="RUNTIME_ARG",
        help="-- 后的参数会原样传给所选 runtime",
    )

    status = subparsers.add_parser(
        "status",
        help="查看服务状态和实际监听端口",
        description="查看当前用户启动的 vLLM 服务状态和实际监听端口。",
    )
    _add_service_id(status)
    status.add_argument(
        "--json",
        action="store_true",
        help="输出机器可读的 JSON",
    )

    stop = subparsers.add_parser(
        "stop",
        help="停止服务并清理本次运行状态",
        description="停止当前用户启动的 vLLM 服务并清理本次运行状态。",
    )
    _add_service_id(stop)
    stop.add_argument(
        "--request-id",
        type=_request_id,
        help="只停止 request ID 仍匹配的运行",
    )

    subparsers.add_parser("submit")

    run = subparsers.add_parser("run")
    _add_service_id(run)

    cleanup = subparsers.add_parser("cleanup")
    _add_service_id(cleanup)

    status_internal = subparsers.add_parser("status-internal")
    _add_service_id(status_internal)
    status_internal.add_argument("--json", action="store_true")

    stop_internal = subparsers.add_parser("stop-internal")
    _add_service_id(stop_internal)
    stop_internal.add_argument("--request-id", type=_request_id)
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
        policy = AutoPortPolicy(
            mode="auto",
            range_start=start,
            range_end=end,
        )
    if not args.vllm_argv or args.vllm_argv[0] != "--":
        raise CliError("runtime argv must follow --")
    vllm_argv = args.vllm_argv[1:]
    if not vllm_argv:
        raise CliError("runtime argv must follow --")
    candidate_payload: dict[str, object] = {
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
        "vllm_argv": vllm_argv,
    }
    if args.output is not None:
        candidate_payload["output"] = str(args.output)
    return CandidateRequest.from_dict(candidate_payload)


def _launcher_path() -> str:
    return str(Path(sys.argv[0]).resolve())


def _run_privileged(arguments: list[str], *, input_text: str | None = None) -> int:
    result = subprocess.run(
        ["/usr/bin/sudo", _launcher_path(), *arguments],
        input=input_text,
        text=True,
        check=False,
    )
    return result.returncode


def _require_root() -> None:
    if os.geteuid() != 0:
        raise CliError("internal command requires root")


def _requester() -> tuple[int, bool]:
    _require_root()
    if "SUDO_UID" not in os.environ:
        return 0, True
    caller = caller_from_sudo_environment(
        os.environ,
        effective_uid=os.geteuid(),
    )
    return caller.uid, False


def _print_status(payload: dict[str, object], *, as_json: bool) -> None:
    if as_json:
        json.dump(payload, sys.stdout, ensure_ascii=False, sort_keys=True)
        sys.stdout.write("\n")
        return
    endpoint = ""
    if "port" in payload:
        endpoint = f" endpoint={payload['scrape_host']}:{payload['port']}"
    systemd = payload["systemd"]
    if not isinstance(systemd, dict):
        raise CliError("invalid systemd status")
    print(
        f"{payload['service_id']} state={payload['state']} "
        f"systemd={systemd['ActiveState']}/{systemd['SubState']}"
        f"{endpoint}"
    )


def _handle_submit(
    layout: RuntimeLayout,
    systemd: SystemdSubprocessController,
) -> int:
    _require_root()
    caller = caller_from_sudo_environment(
        os.environ,
        effective_uid=os.geteuid(),
    )
    candidate = CandidateRequest.from_dict(json.load(sys.stdin))
    request = submit_request(
        layout,
        candidate,
        caller,
        systemd=systemd,
        request_id_factory=lambda: uuid.uuid4().hex,
    )
    json.dump(
        {
            "service_id": request.service_id,
            "request_id": request.request_id,
            "state": "accepted",
        },
        sys.stdout,
        sort_keys=True,
    )
    sys.stdout.write("\n")
    return 0


def _handle_status_internal(
    args: argparse.Namespace,
    layout: RuntimeLayout,
    systemd: SystemdSubprocessController,
) -> int:
    requester_uid, administrator = _requester()
    payload = service_status(
        layout,
        args.service_id,
        requester_uid=requester_uid,
        administrator=administrator,
        systemd=systemd,
    )
    _print_status(payload, as_json=args.json)
    return 0


def _handle_stop_internal(
    args: argparse.Namespace,
    layout: RuntimeLayout,
    systemd: SystemdSubprocessController,
) -> int:
    requester_uid, administrator = _requester()
    stop_service(
        layout,
        args.service_id,
        requester_uid=requester_uid,
        administrator=administrator,
        systemd=systemd,
        expected_request_id=args.request_id,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(argv) if argv is not None else sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(arguments)
    layout = RuntimeLayout.system()
    systemd = SystemdSubprocessController()
    try:
        if args.command == "start":
            if os.geteuid() == 0:
                raise CliError("start must be run by a non-root user")
            if "--" not in arguments:
                raise CliError("start requires -- before runtime argv")
            candidate = _candidate_from_args(args)
            return _run_privileged(
                ["submit"],
                input_text=json.dumps(candidate.to_dict()),
            )
        if args.command == "status":
            if os.geteuid() == 0:
                return _handle_status_internal(
                    args,
                    layout,
                    systemd,
                )
            command = ["status-internal", "--service-id", args.service_id]
            if args.json:
                command.append("--json")
            return _run_privileged(command)
        if args.command == "stop":
            if os.geteuid() == 0:
                return _handle_stop_internal(args, layout, systemd)
            command = ["stop-internal", "--service-id", args.service_id]
            if args.request_id is not None:
                command.extend(["--request-id", args.request_id])
            return _run_privileged(command)
        if args.command == "submit":
            return _handle_submit(layout, systemd)
        if args.command == "run":
            _require_root()
            invocation_id = os.environ.get("INVOCATION_ID")
            if invocation_id is None:
                raise CliError("INVOCATION_ID is required")
            run_service(
                layout,
                args.service_id,
                invocation_id=invocation_id,
                pid=os.getpid(),
                dropper=drop_privileges,
                environment_builder=build_runtime_environment,
                change_directory=os.chdir,
                executor=os.execve,
                output_redirector=redirect_output,
            )
            return 0
        if args.command == "cleanup":
            _require_root()
            invocation_id = os.environ.get("INVOCATION_ID")
            if invocation_id is None:
                raise CliError("INVOCATION_ID is required")
            cleanup_invocation(
                layout,
                args.service_id,
                invocation_id=invocation_id,
            )
            return 0
        if args.command == "status-internal":
            return _handle_status_internal(args, layout, systemd)
        if args.command == "stop-internal":
            return _handle_stop_internal(args, layout, systemd)
        raise CliError("unsupported command")
    except (
        CliError,
        ContractError,
        EndpointError,
        RuntimeEnvironmentError,
        IdentityError,
        ServiceError,
        StateError,
        json.JSONDecodeError,
        OSError,
    ) as exc:
        print(f"vllm-service-launch: {exc}", file=sys.stderr)
        return 1
