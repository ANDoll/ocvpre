"""模块 D：一致性校验层（多版本交叉验证）

对应方案文档「四、模块 D：一致性校验层」。
对比原始版本与各还原版本的七维概率向量，计算规避嫌疑分。
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

from .config import ConsistencyConfig
from .schemas import (ConsistencyResult, PerceptionResult, RestoredVersion,
                      TriggerDetail)


class ConsistencyChecker:
    """一致性校验层。"""

    def __init__(self, cfg: ConsistencyConfig):
        self.cfg = cfg

    def check(self, versions: List[RestoredVersion],
              reports: List[dict | None],
              perception: PerceptionResult) -> ConsistencyResult:
        """对原始 + 还原版本结果进行一致性度量。"""
        if not reports or reports[0] is None:
            return ConsistencyResult()

        # 提取各版本的七维类别分数向量（与原模型 label_map 一致）
        labels, orig_vec = self._extract_scores(reports[0])
        if not labels:
            return ConsistencyResult()

        restored_vecs: List[Tuple[str, np.ndarray]] = []
        for i in range(1, len(reports)):
            if reports[i] is None:
                continue
            _, vec = self._extract_scores(reports[i])
            if vec is not None:
                restored_vecs.append((versions[i].version_id, vec))

        if not restored_vecs:
            return ConsistencyResult(
                anomalies_detected=self._collect_anomalies(perception),
            )

        # D.2 单标签稳定性
        label_jumps, trigger_details = self._label_jumps(
            labels, orig_vec, restored_vecs, versions
        )

        # D.3 全局分布偏移 JSD
        jsd_values = [self._jsd(orig_vec, v) for _, v in restored_vecs]

        # D.4 多视角投票触发标记
        trigger_flag = any(t.flagged for t in trigger_details)

        # D.5 规避置信度评分
        evasion_score = self._evasion_score(
            label_jumps, jsd_values, trigger_flag, perception.anomaly_count
        )

        # D.6 风险分级（传入 trigger_details 让分类不依赖感知层）
        level, evasion_type = self._classify(evasion_score, perception, trigger_details)

        # 证据链
        evidence = self._build_evidence(label_jumps, trigger_details, perception)

        return ConsistencyResult(
            evasion_score=float(evasion_score),
            evasion_level=level,
            evasion_type=evasion_type,
            anomalies_detected=self._collect_anomalies(perception),
            trigger_details=trigger_details,
            evidence_chain=evidence,
            label_jumps={k: float(v) for k, v in label_jumps.items()},
            jsd_values=[float(j) for j in jsd_values],
            max_jsd=float(max(jsd_values)) if jsd_values else 0.0,
        )

    # ─────────────── 分数提取 ───────────────

    @staticmethod
    def _extract_scores(report: dict | None) -> Tuple[List[str], np.ndarray | None]:
        """从 /detect 响应中提取七维类别分数向量。

        重要设计原则（与 fusion.py 保持一致）：
        - category_scores 是类别分数的**唯一来源**
        - harmful_segments[].score **不投影到标签**（segment 的 category
          可能是模型误判，投影会放大错误）
        - calibrated_score **不投影到标签**（它是总体违规分，不对应特定类别）
        - 支持长短标签名
        """
        if not report:
            return [], None
        detection = report.get("detection") or report

        # 长标签 → 短标签映射
        long_map = {
            "A person smoking a cigarette": "Smoke",
            "Blood on the ground, bloody scene, gore": "Blood",
            "People fighting and hitting each other": "Violent",
            "Person making aggressive gestures": "Abusive",
            "Sexually suggestive content and exposure": "Sexy",
            "Displaying large amounts of cash suspiciously": "Money",
            "Politically sensitive content and symbols": "Policy",
        }
        labels = ["Smoke", "Blood", "Violent", "Abusive", "Sexy", "Money", "Policy"]
        normalized = {l: 0.0 for l in labels}

        # 只用 category_scores（支持长短标签）
        scores = detection.get("category_scores") or {}
        for key, val in scores.items():
            short = long_map.get(key, key)
            if short in normalized:
                normalized[short] = max(normalized[short], float(val))

        vec = np.array([normalized[l] for l in labels], dtype=np.float64)
        return labels, vec

    # ─────────────── D.2 单标签跃升 ───────────────

    def _label_jumps(self, labels: List[str], orig: np.ndarray,
                     restored: List[Tuple[str, np.ndarray]],
                     versions: List[RestoredVersion]
                     ) -> Tuple[Dict[str, float], List[TriggerDetail]]:
        """计算每个标签的最大跃升幅度 + 触发详情。"""
        jumps: Dict[str, float] = {}
        details: List[TriggerDetail] = []

        for i, label in enumerate(labels):
            orig_score = float(orig[i])
            max_delta = 0.0
            best_version = None
            best_restored = orig_score
            for vid, vec in restored:
                delta = float(vec[i] - orig[i])
                if delta > max_delta:
                    max_delta = delta
                    best_version = vid
                    best_restored = float(vec[i])

            jumps[label] = max_delta

            # 标记触发：跃升 > 阈值 或 断崖式
            flagged = False
            if max_delta > self.cfg.label_jump_threshold:
                flagged = True
            if (max_delta > self.cfg.label_cliff_threshold
                    and best_restored > self.cfg.restored_high_confidence):
                flagged = True

            if best_version and max_delta > 0.01:
                # 找到对应的变换名
                transform_name = self._lookup_transform(versions, best_version)
                details.append(TriggerDetail(
                    transform=transform_name,
                    label=label,
                    original_score=orig_score,
                    restored_score=best_restored,
                    delta=max_delta,
                    flagged=flagged,
                ))

        return jumps, details

    @staticmethod
    def _lookup_transform(versions: List[RestoredVersion], version_id: str) -> str:
        for v in versions:
            if v.version_id == version_id:
                return ",".join(v.transforms_applied) if v.transforms_applied else "unknown"
        return "unknown"

    # ─────────────── D.3 JSD ───────────────

    @staticmethod
    def _jsd(p: np.ndarray, q: np.ndarray) -> float:
        """Jensen-Shannon 散度，上限 ln(2)≈0.693。"""
        p = np.clip(p, 1e-12, 1.0)
        q = np.clip(q, 1e-12, 1.0)
        p = p / p.sum()
        q = q / q.sum()
        m = 0.5 * (p + q)

        def kl(a, b):
            return float(np.sum(a * np.log(a / b)))

        return 0.5 * kl(p, m) + 0.5 * kl(q, m)

    # ─────────────── D.5 规避分 ───────────────

    def _evasion_score(self, label_jumps: Dict[str, float],
                       jsd_values: List[float], trigger_flag: bool,
                       anomaly_count: int) -> float:
        w1, w2, w3, w4 = self.cfg.evasion_weights
        max_jump = max(label_jumps.values()) if label_jumps else 0.0
        mean_jsd = float(np.mean(jsd_values)) if jsd_values else 0.0
        trigger = 1.0 if trigger_flag else 0.0
        # 异常数归一化（最多 5 类）
        anomaly_norm = min(1.0, anomaly_count / 5.0)

        score = (w1 * max_jump + w2 * mean_jsd / math.log(2)
                 + w3 * trigger + w4 * anomaly_norm)
        return float(min(1.0, max(0.0, score)))

    # ─────────────── D.6 分级 ───────────────

    def _classify(self, score: float, perception: PerceptionResult,
                  details: List[TriggerDetail] | None = None
                  ) -> Tuple[str, str | None]:
        if score >= self.cfg.high_risk_threshold:
            level = "high"
        elif score >= self.cfg.medium_risk_threshold:
            level = "medium"
        elif score >= self.cfg.low_risk_threshold:
            level = "low"
        else:
            level = "none"

        # 规避类型识别（不依赖感知层，根据 trigger_details 实际触发的变换推断）
        # 这让保险镜像还原机制生效：即使感知层未检测到镜像异常，
        # 只要 mirror 变换后分数显著跃升，就识别为镜像规避
        evasion_type = self._infer_evasion_type(details, perception)
        return level, evasion_type

    @staticmethod
    def _infer_evasion_type(details: List[TriggerDetail] | None,
                            perception: PerceptionResult) -> str | None:
        """根据实际触发的变换推断规避类型，而非依赖感知层的标记。

        优先级：
        1. 如果有 flagged 的 trigger_detail，取其变换对应的规避类型
        2. 否则回退到感知层的异常标记
        3. 都没有则返回 None
        """
        if details:
            # 找到 delta 最大且 flagged 的项
            flagged = [d for d in details if d.flagged]
            candidates = flagged or details
            best = max(candidates, key=lambda d: d.delta)
            if best.delta > 0.01:
                # 变换名 → 规避类型映射
                transform_lower = best.transform.lower()
                if "vflip" in transform_lower:
                    return "vertical_flip"
                if "mirror" in transform_lower:
                    return "mirror_flip"
                if "crop" in transform_lower:
                    return "border_occlusion"
                if "brighten" in transform_lower or "denoise" in transform_lower:
                    return "quality_degradation"
                if "split" in transform_lower:
                    return "split_screen"
                if "resample" in transform_lower:
                    return "speed_change"

        # 回退到感知层标记
        if perception.is_vertical_flip_suspicious:
            return "vertical_flip"
        if perception.is_mirror_suspicious:
            return "mirror_flip"
        if perception.is_border_occluded:
            return "border_occlusion"
        if perception.is_quality_degraded or perception.is_darkened:
            return "quality_degradation"
        if perception.is_split_screen:
            return "split_screen"
        if perception.is_speed_abnormal:
            return "speed_change"
        if perception.is_flickering:
            return "flicker_insertion"
        return None

    # ─────────────── 异常收集 ───────────────

    @staticmethod
    def _collect_anomalies(p: PerceptionResult) -> List[str]:
        out = []
        if p.is_mirror_suspicious:
            out.append("mirror_symmetry")
        if p.is_vertical_flip_suspicious:
            out.append("vertical_flip")
        if p.is_border_occluded:
            out.append("border_occlusion")
        if p.is_quality_degraded:
            out.append("quality_degraded")
        if p.is_split_screen:
            out.append("split_screen")
        if p.is_speed_abnormal:
            out.append("speed_abnormal")
        if p.is_darkened:
            out.append("darkened")
        if p.is_flickering:
            out.append("flicker")
        return out

    # ─────────────── 证据链 ───────────────

    def _build_evidence(self, label_jumps: Dict[str, float],
                        details: List[TriggerDetail],
                        perception: PerceptionResult) -> str:
        if not details:
            return "未检测到显著规避特征"
        # 取跃升幅度最大的项
        top = max(details, key=lambda d: d.delta)
        evidence = (
            f"检测到变换【{top.transform}】后 {top.label} 标签置信度"
            f"从 {top.original_score:.2f} 跃升至 {top.restored_score:.2f}，"
            f"跃升幅度 {top.delta:.2f}"
        )
        if perception.is_mirror_suspicious:
            evidence += "；疑似镜像规避上传"
        if perception.is_vertical_flip_suspicious:
            evidence += "；疑似上下翻转规避"
        if perception.is_border_occluded:
            evidence += "；疑似边框遮挡"
        if perception.is_quality_degraded or perception.is_darkened:
            evidence += "；疑似画质降质/蒙版遮挡"
        if perception.is_flickering:
            evidence += "；疑似闪烁插入规避"
        return evidence
