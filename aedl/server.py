"""AEDL FastAPI 服务入口

对外提供端点：
- POST /api/v1/aedl/detect   完整 AEDL 流程（感知→还原→送审→校验→融合）
- POST /api/v1/aedl/preview  还原预览（仅感知+变换，不调后端，用于测试验证）
- GET  /api/v1/aedl/temp/{f} 下载还原视频文件
- GET  /api/v1/aedl/health   健康检查

启动：
    python -m aedl.server
    或
    uvicorn aedl.server:app --host 0.0.0.0 --port 8001 --reload
"""
from __future__ import annotations

import os
import uuid
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import load_config
from .pipeline import AEDLPipeline
from .schemas import AEDLError, AEDLResponse, PreviewResponse

SUPPORTED_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}

app = FastAPI(
    title="AEDL - 对抗规避检测层",
    description="部署于 NSFW Detector 上游的中间模块，检测并还原视频规避手段。",
    version="1.0.0",
)

cfg = load_config()
os.makedirs(cfg.server.temp_dir, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg.server.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = AEDLPipeline(cfg)

# 挂载静态文件：让还原视频可被下载
# 访问路径：/api/v1/aedl/temp/{filename}
app.mount(
    "/api/v1/aedl/temp",
    StaticFiles(directory=cfg.server.temp_dir),
    name="aedl-temp",
)


@app.get("/")
async def root():
    return {"name": "AEDL", "version": "1.0.0", "docs": "/docs"}


@app.get("/api/v1/aedl/health")
async def health():
    backend_ok = await pipeline.backend.health()
    return {
        "status": "ok" if backend_ok else "degraded",
        "backend_available": backend_ok,
        "backend_url": cfg.backend.base_url,
        "temp_dir": cfg.server.temp_dir,
    }


# ───────────────────────── 完整检测端点 ─────────────────────────

@app.post(
    "/api/v1/aedl/detect",
    response_model=AEDLResponse,
    responses={400: {"model": AEDLError}, 500: {"model": AEDLError}, 502: {"model": AEDLError}},
    summary="对抗规避检测：上传视频并执行完整 AEDL 流程",
    description="""
**响应结构兼容原 `/api/v1/detect`**：

顶层字段保持原 NSFW Detector 的 `detection` + `report` 结构不变，
前端原有渲染代码（`data.detection.xxx` / `data.report.xxx`）**无需任何修改**。

新增可选字段 `aedl_analysis`，前端可渐进式接入规避分析展示：
- 不渲染该字段不影响原有检测结果的展示
- 渲染该字段可展示规避类型、证据链、多版本对比等

**请求参数与原 `/api/v1/detect` 完全一致**（仅端点路径不同）：
- `file`: 视频文件
- `threshold`: 临时异常阈值（可选）

新增可选参数：
- `uploader`: 上传者标识（用于白名单匹配）
- `keep_temp`: 保留还原视频用于调试（默认 false）
    """,
)
async def detect(
    file: UploadFile = File(..., description="视频文件"),
    threshold: Optional[float] = Form(None, description="临时异常阈值 [0,1]"),
    uploader: Optional[str] = Form(None, description="上传者标识（用于白名单）"),
    keep_temp: bool = Form(False, description="保留还原视频用于调试"),
):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in SUPPORTED_VIDEO_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的视频格式: {ext}，支持: {sorted(SUPPORTED_VIDEO_EXT)}",
        )

    contents = await file.read()
    max_bytes = cfg.server.max_upload_size_mb * 1024 * 1024
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"文件过大: {len(contents) / 1024 / 1024:.1f}MB，上限 {cfg.server.max_upload_size_mb}MB",
        )

    tmp_path = os.path.join(cfg.server.temp_dir, f"input_{uuid.uuid4().hex[:8]}{ext}")
    try:
        with open(tmp_path, "wb") as f:
            f.write(contents)

        response = await pipeline.detect(
            tmp_path, threshold=threshold, uploader=uploader, keep_temp=keep_temp
        )
        return response

    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AEDL 处理失败: {e}")
    finally:
        # 上传的原始临时文件始终清理（还原文件由 keep_temp 控制）
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass


# ───────────────────────── 还原预览端点（测试用） ─────────────────────────

@app.post(
    "/api/v1/aedl/preview",
    response_model=PreviewResponse,
    responses={400: {"model": AEDLError}, 500: {"model": AEDLError}},
    summary="【测试用】还原预览：仅感知+变换，不调用后端模型",
    description="""
**专门用于独立测试 AEDL 还原功能**。

输入一个视频（如镜像翻转的视频），返回：
- 输入感知结果（检测到的异常类型）
- 策略路由决策（触发了哪些变换）
- 原始 + 各还原版本的下载 URL

**不调用后端 NSFW Detector**，仅验证「感知 + 变换还原」是否符合预期。

**典型测试场景**：
1. 用 `cv2.flip(video, 1)` 制作一个镜像翻转的视频
2. 上传到 `/api/v1/aedl/preview`
3. 响应中应包含 `mirror` 类型的还原版本
4. 下载还原视频，对比是否还原成了正常方向

下载还原视频：访问 `video_url` 字段返回的路径即可，如：
`GET http://localhost:8001/api/v1/aedl/temp/restored_1_xxxx.mp4`
    """,
)
async def preview(
    file: UploadFile = File(..., description="视频文件"),
    uploader: Optional[str] = Form(None, description="上传者标识（用于白名单）"),
):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in SUPPORTED_VIDEO_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的视频格式: {ext}，支持: {sorted(SUPPORTED_VIDEO_EXT)}",
        )

    contents = await file.read()
    max_bytes = cfg.server.max_upload_size_mb * 1024 * 1024
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"文件过大: {len(contents) / 1024 / 1024:.1f}MB，上限 {cfg.server.max_upload_size_mb}MB",
        )

    # 预览端点保留输入文件，方便对比（用 preview_input_ 前缀区分）
    tmp_path = os.path.join(cfg.server.temp_dir, f"preview_input_{uuid.uuid4().hex[:8]}{ext}")
    try:
        with open(tmp_path, "wb") as f:
            f.write(contents)

        response = await pipeline.preview(tmp_path, uploader=uploader)
        return response

    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AEDL 预览失败: {e}")


# ───────────────────────── 临时文件管理 ─────────────────────────

@app.delete(
    "/api/v1/aedl/temp",
    summary="清理所有临时还原视频文件",
    description="清理 temp_dir 下的所有还原视频文件，释放磁盘空间。",
)
async def cleanup_temp(max_age_hours: int = 0):
    """清理临时还原视频文件。max_age_hours=0 表示清理全部。"""
    if max_age_hours == 0:
        # 立即清理全部
        removed = 0
        if os.path.exists(cfg.server.temp_dir):
            for fname in os.listdir(cfg.server.temp_dir):
                fpath = os.path.join(cfg.server.temp_dir, fname)
                if os.path.isfile(fpath):
                    try:
                        os.remove(fpath)
                        removed += 1
                    except OSError:
                        pass
        return {"removed": removed, "message": "已清理所有临时文件"}
    else:
        removed = pipeline.cleanup_old_temp(max_age_hours)
        return {"removed": removed, "max_age_hours": max_age_hours}


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "HTTPException", "detail": exc.detail},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "aedl.server:app",
        host=cfg.server.host,
        port=cfg.server.port,
        reload=False,
    )
