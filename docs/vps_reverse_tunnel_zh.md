# G1 通过 VPS 反向隧道远程访问（中文说明）

本文档记录当前已配置完成的方案：`G1 -> 阿里云 VPS` 反向 SSH 隧道。

## 1. 当前已生效配置

- VPS：`47.243.212.218`
- G1 systemd 服务：`g1-revtunnel.service`
- 服务状态：已 `enabled`（开机自启）且 `active`（正在运行）
- 隧道映射：
  - `VPS:2222 -> G1:localhost:22`（SSH 登录 G1）
  - `VPS:18091 -> G1:192.168.123.164:8091`（HTTP 转发）
  - `VPS:18080 -> G1:192.168.123.164:8080`（HTTP 转发）

## 2. 如何连接

### 2.1 从任意外网电脑 SSH 到 G1

```bash
ssh -p 2222 unitree@47.243.212.218
```

### 2.2 从任意外网电脑访问 G1 的 Web 服务

- `http://47.243.212.218:18091/`
- `http://47.243.212.218:18080/`

说明：只有当 G1 本机对应端口（`8091` / `8080`）有进程监听时，网页才会打开。

## 3. 会不会自动启动？

会。当前配置是 **开机自动启动**：

- `systemctl is-enabled g1-revtunnel.service` 返回 `enabled`
- `systemctl is-active g1-revtunnel.service` 返回 `active`

## 4. 常用运维命令（在 G1 上执行）

```bash
# 查看状态
sudo systemctl status g1-revtunnel.service

# 重启隧道
sudo systemctl restart g1-revtunnel.service

# 停止隧道
sudo systemctl stop g1-revtunnel.service

# 取消开机自启
sudo systemctl disable g1-revtunnel.service

# 查看日志
journalctl -u g1-revtunnel.service -n 100 --no-pager
```

## 5. 云侧放行要求（阿里云）

安全组入站至少放行以下 TCP 端口：

- `22`（VPS 自己 SSH，G1 连接 VPS 需要）
- `2222`（外网 SSH 到 G1）
- `18091`（外网访问 G1:8091）
- `18080`（外网访问 G1:8080）

建议将来源 IP 收敛到你的常用公网地址，避免全网开放。

## 6. 故障排查

### 6.1 SSH 超时

1. 在 G1 看服务状态：

```bash
sudo systemctl status g1-revtunnel.service
```

2. 在 VPS 看端口是否监听：

```bash
ss -ltnp | grep -E ':2222|:18091|:18080'
```

3. 检查阿里云安全组是否放行对应 TCP 端口。

### 6.2 网页打不开

先确认 G1 本机业务进程已监听：

```bash
ss -ltnp | grep -E ':8080|:8091'
```

如果没有监听，需要先启动发布这些端口的服务。

## 7. 已移除的 Tailscale 说明

本机已卸载 Tailscale 及相关配置，后续默认使用 VPS 反向隧道方案。
