# AEDL 启动指南

## 环境要求

### 基础环境

- **操作系统**：Windows / Linux / macOS
- **Python**：3.10+
- **磁盘空间**：≥ 1GB（含 CLIP 模型缓存约 335MB）

### 后端依赖

AEDL 需要原 NSFW Detector 服务作为下游推理后端，启动前请确认：

- NSFW Detector 服务已启动，默认地址 `http://localhost:8000`
- 提供 `POST /api/v1/detect` 接口可供调用

如后端地址非默认值，请在 [configs/aedl.yaml](configs/aedl.yaml) 中修改 `backend.base_url`，或设置环境变量 `AEDL_BACKEND_URL`。

### CLIP 依赖（屏中屏模式专属）

CLIP re-rank 复用网页模型的本地 `clip` 包，启动前请确认：

- `clip` 包路径：`D:/nsfwtest/NFSW_Detector/clip`（Windows 默认）
- 首次启动会自动下载 CLIP ViT-B/16 权重（约 335MB）

如路径非默认值，请在 [configs/aedl.yaml](configs/aedl.yaml) 中修改 `clip_rerank.clip_module_path`。

> CLIP re-rank 为可选项，加载失败时自动降级为纯"按涨幅精准放大"方案，不影响 AEDL 其他功能。

### Python 依赖

安装依赖：

```bash
pip install -r aedl/requirements.txt
```

依赖清单见 [aedl/requirements.txt](aedl/requirements.txt)，主要包括：

| 库 | 用途 |
|----|------|
| fastapi + uvicorn | Web 服务 |
| httpx | 调用原 `/api/v1/detect` |
| opencv-python | CV 算子 |
| numpy | 数值计算 |
| pydantic | 数据校验 |
| pyyaml | 配置加载 |
| torch + pillow + packaging | CLIP re-rank（可选） |

---

## 如何启动

### 1. 修改配置（可选）

编辑 [configs/aedl.yaml](configs/aedl.yaml) 调整关键参数：

```yaml
backend:
  base_url: http://localhost:8000   # 后端 NSFW Detector 地址
  timeout: 120                        # 推理超时（秒）

server:
  host: 0.0.0.0
  port: 8001                          # AEDL 服务端口

clip_rerank:
  enabled: true                       # 是否启用 CLIP re-rank
  clip_module_path: "D:/nsfwtest/NFSW_Detector/clip"
```

也可通过环境变量覆盖：

| 环境变量 | 说明 | 默认值 |
|----------|------|--------|
| `AEDL_CONFIG` | 配置文件路径 | `configs/aedl.yaml` |
| `AEDL_BACKEND_URL` | 后端 NSFW Detector 地址 | `http://localhost:8000` |
| `AEDL_PORT` | AEDL 服务端口 | `8001` |

### 2. 启动服务

**方式一：直接运行**

```bash
python -m aedl.server
```

**方式二：开发模式（热重载）**

```bash
uvicorn aedl.server:app --host 0.0.0.0 --port 8001 --reload
```

### 3. 验证启动

启动成功后，访问以下地址确认服务可用：

- 健康检查：`http://localhost:8001/api/v1/aedl/health`
- Swagger UI：`http://localhost:8001/docs`

### 4. 调用接口

```bash
curl -X POST http://localhost:8001/api/v1/aedl/detect \
  -F "file=@test.mp4" \
  -F "threshold=0.5"
```

详细接口规范见 [README.md](README.md)。

### 5. 更新代码后重启

修改 AEDL 源码后，需清除 `__pycache__` 并重启服务，确保新代码生效：

```bash
# 清除字节码缓存（Windows PowerShell）
Get-ChildItem -Path . -Filter __pycache__ -Recurse -Directory | Remove-Item -Recurse -Force

# 清除字节码缓存（Linux/macOS）
find . -type d -name __pycache__ -exec rm -rf {} +

# 重启服务
python -m aedl.server
```
