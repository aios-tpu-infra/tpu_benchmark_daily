# Vendored vLLM service launcher

## TL;DR

`vllm-service-launch` 不再依赖 root、sudo 或 init system。每个 vLLM 服务对应一个
轻量 supervisor；supervisor 在后台创建并监控独立 process group，发布 runtime metadata
与 Prometheus file-SD target，服务结束后删除两者并自行退出。server 与 supervisor 默认
继承调用方 stdout/stderr，因此从 Pod 主启动流程调用时日志可由容器日志系统直接采集。

仓库入口可开箱即用：

```bash
vendor/vllm-service-launch/bin/vllm-service-launch --help
```

也可部署到 `/usr/local`：

```bash
sudo scripts/install_vllm_service_launcher.sh
/usr/local/bin/vllm-service-launch --help
```

sudo 仅用于写 `/usr/local`；安装完成后的 `start/status/stop` 不提权。

## 运行模型

公共命令只有：

```text
vllm-service-launch start
vllm-service-launch status
vllm-service-launch stop
```

`start` 先写私有 request，通过一次性 pipe 等待 supervisor 完成 server spawn、runtime
metadata 和 Prometheus target 发布，然后返回。返回 0 表示 supervisor 已接管进程，不表示
模型加载或 HTTP `/health` 已 Ready。

supervisor 记录自身和 server 的 PID 及 `/proc/<pid>/stat` start time。signal 前校验对应
身份，以免 PID 复用时误杀无关进程。server 正常退出、被 TERM/INT/KILL，或 leader 退出
但 worker 仍存活时，存活的 supervisor 作为运行实例锚点收敛整个 process group，删除
metadata 后退出。

若 supervisor 被 KILL 而 server 仍在，`status` 报告 `orphaned`，`stop` 直接收敛已验证的
server process group。若二者同时被 KILL，当前没有进程能执行 finally；下一次
`start/status/stop` 会根据 PID/start time 清理 stale metadata。

有一个刻意保守的安全边界：若 supervisor 和 server leader 都已消失，但系统中仍出现
相同数值 PGID/session 的成员，launcher 无法区分“原 worker”与容器/PID namespace 重建后
复用 ID 的无关进程。此时 `status/stop/start` 会保留 metadata 并报错，不会对裸 PGID
发送信号。先在对应 PID namespace 内人工核验成员，再清除进程或 metadata；安全性优先于
自动清理这一极端 orphan。

若 `start` 在 supervisor 完成原子 claim 前被强制杀死，未认领 request 会保留，避免
并发 `status/start` 误删即将接管的 child；执行一次 `stop` 会原子标记取消并清理它。

## 状态与 Prometheus target

私有 state 默认为当前工作目录：

```text
$PWD/.state/vllm-service-launch/
├── registry.lock
└── services/<service-id>/
    ├── state.lock
    ├── request.json
    └── runtime.json
```

daily 脚本显式使用项目内 `.state/vllm-service-launch`。环境变量值不会写入 metadata。
`state.lock` 是同一 service ID 长期复用的协调 inode，服务结束后保留；request、runtime
和公共 target 会被删除。

Prometheus target 默认写入：

```text
/run/vllm-metrics-targets/targets/<service-id>.json
```

这不是 Prometheus 默认路径，而是当前 Prometheus Agent 的 file-SD 配置契约。launcher
不会在运行时创建或修改该系统目录的权限。部署阶段必须让 runtime user 可写，例如主机
部署可执行：

```bash
sudo install -d -o "$(id -u)" -g "$(id -g)" -m 0755 \
  /run/vllm-metrics-targets/targets
```

容器部署应使用 init container、`emptyDir`/`fsGroup` 或等价机制创建并授权目录；
Prometheus Agent 继续只读挂载。不要使用 `0777`。测试环境可以通过 `--target-root` 或
`VLLM_SERVICE_TARGET_ROOT` 指向临时目录。

## 开箱即用与安装版

daily 默认固定使用仓库版本，避免 PATH 中的旧安装版静默覆盖：

```bash
scripts/start_prefill_server.sh --config dp8
```

显式验证安装版：

```bash
VLLM_SERVICE_LAUNCH=/usr/local/bin/vllm-service-launch \
  scripts/start_prefill_server.sh --config dp8
```

安装脚本只复制 executable 和 Python library。它支持不改系统的 staging：

```bash
stage_root=$(mktemp -d)
scripts/install_vllm_service_launcher.sh --root "$stage_root"
python3.12 "$stage_root/usr/local/bin/vllm-service-launch" --help
find "$stage_root" -type f -print
```

预期不存在 `/etc`、`/run`、sudoers、unit 或 tmpfiles 产物。

从旧 systemd 版升级时，新安装器会在发现以下旧资产后停止，不会静默覆盖：

```text
/etc/sudoers.d/vllm-service-launch
/etc/systemd/system/vllm@.service
/usr/lib/tmpfiles.d/vllm-metrics-targets.conf
```

迁移时先逐个停止 `systemctl list-units 'vllm@*.service'` 列出的旧实例，确认没有仍需保留
的 server，再执行：

```bash
sudo rm -f /etc/sudoers.d/vllm-service-launch \
  /etc/systemd/system/vllm@.service \
  /usr/lib/tmpfiles.d/vllm-metrics-targets.conf
sudo systemctl daemon-reload
sudo scripts/install_vllm_service_launcher.sh
```

必须删除旧 tmpfiles 文件，否则它可能在重启后重新创建 root-owned `/run` 目录。安装器会
删除旧 Python library 中已废弃且不再 import 的 `identity.py`，不会删除
`/run/vllm-metrics-targets/targets` 及其中由其他服务管理的 target。

## 环境、附加参数和日志

supervisor 以调用方完整环境作为 uv/Conda activation 基础，因此新 feature env 不需要加入
daily 白名单。通用 CLI 仍支持显式 `--env-file` overlay，但 daily 不再生成 env 文件。

三个服务入口都在 `--` 后原样追加 vLLM 参数：

```bash
scripts/start_dp_decode_server.sh -- --enable-feature-x
scripts/start_prefill_server.sh --config dp8 -- --enable-feature-x
scripts/start_prefill_server.sh --config pcp8 -- --enable-feature-x
```

附加参数位于 benchmark 默认参数之后；重复参数如何解释由 vLLM parser 决定。

默认不关闭或重定向 fd 1/2。由 Pod 主启动流程运行 daily 时，supervisor 和 vLLM 日志进入
Pod stdout/stderr。通过临时 `kubectl exec` 启动时继承的是 exec session fd，不能据此保证
日志进入 Pod 主日志。调用方显式传 `--output ABSOLUTE_PATH` 时，server stdout/stderr 才
改写到该文件。

## 复现与验证

不需要 TPU 的完整 fake-server lifecycle：

```bash
cd /mnt/data/tpu_benchmark_daily
python3.12 -m unittest tests.test_vllm_service_launcher -v
```

该测试覆盖完整环境、argv/stdout、start/status/stop、server KILL、orphan、PID/PGID 复用
保护、worker process group、激活取消、stale restart、target 回滚、旧安装检测以及
vendored/staged-installed 两种入口。

运行全部仓库测试与静态检查：

```bash
python3.12 -m unittest discover -s tests -v
bash -n scripts/*.sh
rg -n 'systemctl|journalctl|INVOCATION_ID|vllm@\.service|write_launcher_env' \
  scripts tests vendor/vllm-service-launch README.md
```

真实 TPU 验收还需分别启动 DP decode、DP prefill、PCP prefill，检查 `/health`、Pod stdout
以及 target 创建/删除；fake-server 结果不能外推为硬件 E2E 已通过。
