"""AEDL 主流程编排：串联模块 A→B→C→后端→D→E。

对应方案文档「五、融合决策策略」。
"""
from __future__ import annotations

import os
import time
from typing import List, Optional

from .backend_client import BackendClient
from .config import AEDLConfig
from .consistency import ConsistencyChecker
from .fusion import ResultFusion
from .input_perception import InputPerception
from .schemas import (AEDLResponse, PerceptionResult, PreviewResponse,
                      PreviewVersion, RestoredVersion, RoutingDecision)
from .strategy_router import StrategyRouter
from .transforms import TransformRestorer


class AEDLPipeline:
    """对抗规避检测层主管线。"""

    def __init__(self, cfg: AEDLConfig):
        self.cfg = cfg
        self.perception = InputPerception(cfg.perception)
        self.router = StrategyRouter(cfg.router)
        self.restorer = TransformRestorer(cfg.transform, cfg.server.temp_dir)
        self.backend = BackendClient(cfg.backend)
        self.consistency = ConsistencyChecker(cfg.consistency)
        self.fusion = ResultFusion(cfg.consistency)

    async def detect(self, video_path: str,
                     threshold: float | None = None,
                     uploader: str | None = None,
                     keep_temp: bool = False) -> AEDLResponse:
        """端到端处理：感知 → 路由 → 变换 → 送审 → 校验 → 融合。"""
        t_start = time.time()
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        # 1. 模块 A：输入感知
        perception_result = self.perception.analyze(video_path)

        # 2. 模块 B：策略路由
        routing = self.router.route(perception_result, uploader)

        # 3. 模块 C：变换还原（生成原始 + 还原版本）
        versions, transform_ms = self.restorer.build_versions(video_path, routing)

        # 4. 后端并行送审
        reports, inference_ms = await self.backend.detect_all(
            [v.video_path for v in versions], threshold
        )

        # 5. 模块 D：一致性校验
        consistency_result = self.consistency.check(versions, reports, perception_result)

        # 6. 模块 E：结果融合
        timing = {
            "total": (time.time() - t_start) * 1000,
            "perception": perception_result.check_time_ms,
            "transform": transform_ms,
            "inference": inference_ms,
            "consistency": 0.0,
        }
        response = self.fusion.fuse(versions, reports, consistency_result, timing)

        # 清理临时还原文件（除非用户要求保留）
        if not keep_temp:
            self._cleanup_temp(versions)
        return response

    async def preview(self, video_path: str,
                      uploader: str | None = None) -> PreviewResponse:
        """还原预览（仅感知 + 变换，不调用后端模型）。

        专门用于测试验证：输入一个视频（如镜像翻转的视频），
        返回原始 + 各还原版本的下载 URL，可直接下载查看还原效果。
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        # 1. 模块 A：输入感知
        perception_result = self.perception.analyze(video_path)

        # 2. 模块 B：策略路由
        routing = self.router.route(perception_result, uploader)

        # 3. 模块 C：变换还原（保留文件，供下载验证）
        versions, _ = self.restorer.build_versions(video_path, routing)

        # 4. 构造下载 URL（相对路径，由 server 的静态文件挂载提供）
        preview_versions: List[PreviewVersion] = []
        for v in versions:
            filename = os.path.basename(v.video_path)
            preview_versions.append(PreviewVersion(
                version_id=v.version_id,
                is_original=v.is_original,
                transforms_applied=v.transforms_applied,
                video_url=f"/api/v1/aedl/temp/{filename}",
                video_filename=filename,
            ))

        # 测试建议
        test_hint = self._build_test_hint(perception_result, routing, versions)

        return PreviewResponse(
            perception=perception_result,
            routing_triggered=routing.triggered,
            bypass_reason=routing.bypass_reason,
            versions=preview_versions,
            test_hint=test_hint,
        )

    @staticmethod
    def _build_test_hint(perception: PerceptionResult, routing: RoutingDecision,
                         versions: List[RestoredVersion]) -> str:
        """根据检测结果给出测试建议。"""
        if not routing.triggered:
            return (
                f"未触发变换还原（{routing.bypass_reason or '未知原因'}）。"
                "提示：请上传确实存在规避手段的视频，如左右镜像翻转的视频、"
                "上下加黑边的视频、画质明显降质的视频。"
            )

        transforms_summary = []
        for v in versions:
            if v.is_original:
                continue
            transforms_summary.append(
                f"{v.version_id}: {','.join(v.transforms_applied) if v.transforms_applied else '无变换'}"
            )

        hint_parts = [f"触发了 {len(versions) - 1} 个还原版本："]
        hint_parts.extend(transforms_summary)
        hint_parts.append("请下载各版本视频对比，验证还原效果是否符合预期。")
        hint_parts.append("原始视频作为对照组，还原版本应体现反向操作的效果。")
        return "\n".join(hint_parts)

    def cleanup_old_temp(self, max_age_hours: int = 24) -> int:
        """清理过期的临时还原视频文件。返回清理的文件数。"""
        if not os.path.exists(self.cfg.server.temp_dir):
            return 0
        import time as _time
        now = _time.time()
        count = 0
        for fname in os.listdir(self.cfg.server.temp_dir):
            fpath = os.path.join(self.cfg.server.temp_dir, fname)
            if not os.path.isfile(fpath):
                continue
            age = now - os.path.getmtime(fpath)
            if age > max_age_hours * 3600:
                try:
                    os.remove(fpath)
                    count += 1
                except OSError:
                    pass
        return count

    def _cleanup_temp(self, versions) -> None:
        """删除生成的还原视频文件（保留原始）。"""
        if not self.cfg.server.temp_dir:
            return
        for v in versions:
            if v.is_original:
                continue
            try:
                if os.path.exists(v.video_path):
                    os.remove(v.video_path)
            except OSError:
                pass
