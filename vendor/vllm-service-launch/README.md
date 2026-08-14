# Vendored vLLM service launcher

## TL;DR

这个目录包含 daily benchmark 启动服务所需的完整 launcher 运行闭包。在仓库
根目录执行 `sudo scripts/install_vllm_service_launcher.sh` 即可安装。

## 范围

launcher 入口明确要求 `python3.12`，与本 benchmark 的 Python 版本契约一致。

迁入的最小完整闭包包括：

- `bin/vllm-service-launch` 和 `lib/vllm_service_launch/`；
- `systemd/vllm@.service`；
- `sudoers/vllm-service-launch`；
- `systemd/vllm-metrics-targets.conf`。

Grafana dashboard、Prometheus agent 容器和 AI telemetry collector 没有参与
benchmark 的服务生命周期或结果计算，因此不属于本项目的运行依赖，没有迁入。

## 安装与验证

```bash
sudo scripts/install_vllm_service_launcher.sh
vllm-service-launch --help
systemctl cat 'vllm@.service'
```

安装脚本会复制运行文件、校验并安装 sudoers policy、创建 `/run` 目录，然后执行
`systemd-tmpfiles` 和 `systemctl daemon-reload`。它也支持无系统变更的 staging：

```bash
scripts/install_vllm_service_launcher.sh --root /absolute/staging/root
```

更新 vendored 代码时，应确认核心文件差异，并运行：

```bash
python3.12 -m unittest discover -s tests
```
