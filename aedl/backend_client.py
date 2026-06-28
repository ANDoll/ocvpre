"""后端客户端：调用原有 NSFW Detector 的 /api/v1/detect 端点。

对应 API 文档「3.3 POST /api/v1/detect」。
所有版本（原始 + 还原）通过此客户端并行送审。
"""
from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Dict, List, Tuple

import httpx

from .config import BackendConfig


class BackendClient:
    """调用原有 API 的异步客户端。"""

    def __init__(self, cfg: BackendConfig):
        self.cfg = cfg
        self.base_url = cfg.base_url.rstrip("/")
        self.detect_url = f"{self.base_url}{cfg.detect_endpoint}"

    async def detect_all(self, video_paths: List[str], threshold: float | None = None
                         ) -> Tuple[List[Dict[str, Any] | None], float]:
        """并行检测所有版本。返回 (结果列表, 总耗时 ms)。"""
        t0 = time.time()
        async with httpx.AsyncClient(timeout=self.cfg.timeout) as client:
            tasks = [self._detect_one(client, p, threshold) for p in video_paths]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        out: List[Dict[str, Any] | None] = []
        for r in results:
            if isinstance(r, Exception):
                print(f"[AEDL] backend detect failed: {r}")
                out.append(None)
            else:
                out.append(r)
        return out, (time.time() - t0) * 1000

    async def _detect_one(self, client: httpx.AsyncClient, video_path: str,
                          threshold: float | None) -> Dict[str, Any] | None:
        if not os.path.exists(video_path):
            return None
        with open(video_path, "rb") as f:
            files = {"file": (os.path.basename(video_path), f, "video/mp4")}
            data = {}
            if threshold is not None:
                data["threshold"] = str(threshold)
            for attempt in range(self.cfg.max_retries + 1):
                try:
                    resp = await client.post(self.detect_url, files=files, data=data)
                    if resp.status_code == 200:
                        return resp.json()
                    if resp.status_code in (400, 503):
                        # 客户端错误或服务不可用，不重试
                        print(f"[AEDL] backend {resp.status_code}: {resp.text[:200]}")
                        return None
                    # 5xx 重试
                except (httpx.RequestError, httpx.HTTPStatusError) as e:
                    if attempt == self.cfg.max_retries:
                        print(f"[AEDL] backend retry exhausted: {e}")
                        return None
        return None

    async def health(self) -> bool:
        """检查后端服务是否可用。"""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self.base_url}/api/v1/health")
                return resp.status_code == 200
        except Exception:
            return False
