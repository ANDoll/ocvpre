"""模块 B：策略路由层（异常→变换映射，决策表驱动）

对应方案文档「四、模块 B：策略路由层」。
根据模块 A 的异常信号决策要执行的变换组合。
"""
from __future__ import annotations

import os
from typing import List

from .config import RouterConfig
from .schemas import PerceptionResult, RoutingDecision, TransformSpec


class StrategyRouter:
    """策略路由层：决策表驱动，避免硬编码。"""

    def __init__(self, cfg: RouterConfig):
        self.cfg = cfg

    def route(self, perception: PerceptionResult, uploader: str | None = None) -> RoutingDecision:
        # B.3 白名单旁路（最高优先级）
        if self.cfg.whitelist_enabled and uploader and self._is_whitelisted(uploader):
            return RoutingDecision(triggered=False, bypass_reason="whitelisted")

        # 异常驱动的变换链
        anomaly_chains: List[List[TransformSpec]] = []
        if perception.has_anomaly and perception.anomaly_count > 0:
            anomaly_chains = self._build_chains(perception)

        # 分屏/屏中屏特别处理：额外追加 split_extract+brighten 组合链
        # 因为缩小的有害画面往往伴随亮度不足，单独 split_extract 可能不够，
        # 叠加 brighten 增强亮度，让原模型能更清晰地看到放大的有害画面
        if perception.is_split_screen:
            lines = [{"position": ln.position, "orientation": ln.orientation} for ln in perception.split_lines]
            anomaly_chains.append([TransformSpec(name="split_extract+brighten", params={"lines": lines})])

        # 保险还原变换：无视感知层是否检测到异常，
        # 默认对所有视频追加这些变换链送审。
        # 由模块 D 一致性校验层通过分数对比判断是否真的存在规避。
        # 支持：mirror（水平翻转）、vflip（垂直翻转）、brighten（亮度增强）
        chains: List[List[TransformSpec]] = []
        for transform_name in self.cfg.force_restore_transforms:
            # 检查异常链中是否已经包含该变换（避免重复）
            already_has = any(
                any(s.name == transform_name for s in chain)
                for chain in anomaly_chains
            )
            if not already_has:
                chains.append([TransformSpec(name=transform_name)])

        chains.extend(anomaly_chains)

        if not chains:
            return RoutingDecision(triggered=False, bypass_reason="no_anomaly")

        # 限制还原版本数量
        chains = chains[: self.cfg.max_restored_versions]
        return RoutingDecision(triggered=True, transform_chains=chains)

    # ─────────────── 决策表 ───────────────

    def _build_chains(self, p: PerceptionResult) -> List[List[TransformSpec]]:
        """根据异常→变换决策表生成变换链。"""
        anomalies: List[tuple[str, float, TransformSpec]] = []

        if p.is_mirror_suspicious:
            anomalies.append(("mirror", p.symmetry_score,
                              TransformSpec(name="mirror")))

        if p.is_vertical_flip_suspicious:
            anomalies.append(("vflip", p.vertical_symmetry_score,
                              TransformSpec(name="vflip")))

        if p.is_border_occluded:
            crop_ratio = 1.0 - p.border_width_ratio
            crop_ratio = max(0.80, min(0.95, crop_ratio))
            anomalies.append(("border", p.occlusion_confidence,
                              TransformSpec(name="center_crop", params={"ratio": crop_ratio})))

        if p.is_quality_degraded:
            anomalies.append(("quality", p.blocking_artifact_score,
                              TransformSpec(name="brighten")))
            anomalies.append(("quality_denoise", p.blocking_artifact_score,
                              TransformSpec(name="denoise",
                                            params={"use_gaussian": p.blocking_artifact_score > 0.6})))

        if p.is_darkened:
            # 亮度异常（蒙版/调暗）：用亮度增强还原
            # 置信度 = 1 - 归一化亮度（越暗越需要还原）
            conf = max(0.0, 1.0 - p.mean_brightness / 80.0)
            anomalies.append(("darkness", conf,
                              TransformSpec(name="brighten")))

        if p.is_split_screen:
            lines = [{"position": ln.position, "orientation": ln.orientation} for ln in p.split_lines]
            # 单独 split_extract 变换
            anomalies.append(("split", 1.0,
                              TransformSpec(name="split_extract", params={"lines": lines})))

        if p.is_speed_abnormal:
            anomalies.append(("speed", 1.0,
                              TransformSpec(name="resample",
                                            params={"ratio": p.estimated_speed_ratio})))

        if p.is_flickering:
            # 闪烁：用亮度增强尝试还原有害帧内容
            anomalies.append(("flicker", p.flicker_score,
                              TransformSpec(name="brighten")))

        if not anomalies:
            return []

        # 按置信度降序排序
        anomalies.sort(key=lambda x: x[1], reverse=True)

        # B.2 单一异常 → 单变换链
        if len(anomalies) == 1:
            return self._compose_single(anomalies[0][0], [a[2] for a in anomalies])

        # 叠加异常：按优先级串联，最多 2 项/链
        return self._compose_multi(anomalies)

    def _compose_single(self, name: str, specs: List[TransformSpec]) -> List[List[TransformSpec]]:
        """单一异常的变换组合。"""
        if name == "quality":
            # 画质降质：B + D 串联
            return [[TransformSpec(name="brighten"), TransformSpec(name="denoise")]]
        return [[s] for s in specs[:1]]

    def _compose_multi(self, anomalies: List[tuple[str, float, TransformSpec]]
                       ) -> List[List[TransformSpec]]:
        """多项异常叠加：选置信度最高前 2 项，串联为一条链。"""
        # 取置信度最高的前 2 个异常（去重同类）
        seen = set()
        selected: List[TransformSpec] = []
        for name, conf, spec in anomalies:
            base_name = name.split("_")[0]
            if base_name in seen:
                continue
            seen.add(base_name)
            selected.append(spec)
            if len(selected) >= self.cfg.max_transform_chain:
                break

        if not selected:
            return []

        # 按优先级排序：空间变换 → 画质变换 → 帧率补偿
        priority = {"mirror": 0, "vflip": 0, "center_crop": 1, "split_extract": 2,
                    "brighten": 3, "denoise": 4, "resample": 5}
        selected.sort(key=lambda s: priority.get(s.name, 99))
        return [selected]

    # ─────────────── 白名单 ───────────────

    def _is_whitelisted(self, uploader: str) -> bool:
        path = self.cfg.whitelist_path
        if not os.path.exists(path):
            return False
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and line == uploader:
                    return True
        return False


