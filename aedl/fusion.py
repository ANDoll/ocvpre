"""模块 E：结果融合输出

对应方案文档「四、模块 E：结果融合输出」。

设计原则（方案 A）：
保持原 /api/v1/detect 的 {detection, report} 顶层结构契约不变，
前端原有渲染逻辑零改动。融合后的分数覆盖到原始 detection/report 的对应字段，
规避分析作为 aedl_analysis 可选字段挂载。
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List

from .config import ConsistencyConfig
from .schemas import (AEDLAnalysis, AEDLResponse, ConsistencyResult,
                      ProcessingMetadata, RestoredVersion)


# 原 NSFW Detector 的 7 类标签顺序（与 API.md §8.1 一致）
LABELS = ["Smoke", "Blood", "Violent", "Abusive", "Sexy", "Money", "Policy"]

# 长/短标签映射（原 API 在 is_harmful=true 时返回长描述性标签名）
# 当 category_scores 的 key 无法用短标签匹配时，尝试用长标签匹配
LONG_LABEL_MAP = {
    "A person smoking a cigarette": "Smoke",
    "Blood on the ground, bloody scene, gore": "Blood",
    "People fighting and hitting each other": "Violent",
    "Person making aggressive gestures": "Abusive",
    "Sexually suggestive content and exposure": "Sexy",
    "Displaying large amounts of cash suspiciously": "Money",
    "Politically sensitive content and symbols": "Policy",
}
# 反向映射（短标签 → 长标签）
SHORT_TO_LONG = {v: k for k, v in LONG_LABEL_MAP.items()}

# 预警等级阈值（与 API.md §8.2 一致，左闭右开，HIGH 包含 1.0）
ALERT_THRESHOLDS = [
    ("HIGH", 0.8),
    ("MEDIUM", 0.5),
    ("LOW", 0.3),
    ("SAFE", 0.0),
]


class ResultFusion:
    """结果融合输出层。"""

    def __init__(self, cfg: ConsistencyConfig):
        self.cfg = cfg

    def fuse(self, versions: List[RestoredVersion],
             reports: List[dict | None],
             consistency: ConsistencyResult,
             timing: Dict[str, float]) -> AEDLResponse:
        """E.1 保守最大值 + 保持原 API 契约结构。"""

        # 取原始版本作为基底（保持原 detection/report 字段结构）
        original = reports[0] if reports and reports[0] else {}
        original_detection = copy.deepcopy(original.get("detection") or {})
        original_report = copy.deepcopy(original.get("report") or {})

        # E.1 计算融合后的最终分数
        # 设计原则：还原版本的分数只有在「触发跃升」时才融合
        # —— 触发跃升说明还原后模型看到更多违规内容，应取高分；
        #    未触发说明还原版本未带来新信息，应保留原始版本的真实分数。
        #    这样可避免 center_crop 裁剪后构图变化导致的模型误判污染最终结果。
        triggered_labels = {d.label for d in consistency.trigger_details if d.flagged}
        final_scores = self._fuse_scores(reports, triggered_labels)

        # anomaly_score 用 calibrated_score 兜底（最准确的总体违规分）
        # 这与原 API 的契约一致：anomaly_score 反映总体违规程度，
        # category_scores 反映各类别分数，两者可以独立。
        fused_anomaly = self._fuse_anomaly_score(reports, final_scores, triggered_labels)
        threshold = self._get_threshold(original_detection)
        # predicted_categories：只基于 category_scores >= threshold
        # 这是唯一安全的方案 —— 不继承原 API 的 predicted_categories 误判
        # （原 API 模型可能把血腥/吸烟视频误标为辱骂，取并集会放大这个错误）
        predicted_cats = [l for l in LABELS if final_scores.get(l, 0.0) >= threshold]
        predicted_cats.sort(key=lambda l: final_scores[l], reverse=True)

        original_detection["category_scores"] = final_scores
        original_detection["anomaly_score"] = float(fused_anomaly)
        # is_harmful 用 anomaly_score 判断（包含 calibrated_score 兜底）
        original_detection["is_harmful"] = fused_anomaly >= threshold
        original_detection["predicted_categories"] = predicted_cats

        # 合并所有版本的有害时间段和关键帧 URL（去重）
        original_detection["harmful_segments"] = self._merge_harmful_segments(reports)
        original_detection["keyframe_urls"] = self._merge_keyframe_urls(reports)

        # 用融合分数更新 report 的对应字段
        alert_level = self._alert_level(fused_anomaly)
        original_report["anomaly_score"] = float(fused_anomaly)
        original_report["alert_level"] = alert_level
        original_report["harmful_contents"] = self._merge_harmful_contents(reports)
        original_report["summary"] = self._build_summary(
            original_report.get("summary", ""), fused_anomaly, predicted_cats, consistency
        )
        original_report["action_suggestion"] = self._build_action_suggestion(
            alert_level, consistency
        )

        # 构造 AEDL 规避分析附加字段
        needs_review = consistency.evasion_level in ("medium", "high")
        priority = consistency.evasion_level if consistency.evasion_level != "none" else "none"

        meta = ProcessingMetadata(
            versions_generated=sum(1 for v in versions if not v.is_original),
            model_inference_count=sum(1 for r in reports if r is not None),
            total_processing_time_ms=timing.get("total", 0.0),
            perception_time_ms=timing.get("perception", 0.0),
            transform_time_ms=timing.get("transform", 0.0),
            inference_time_ms=timing.get("inference", 0.0),
            consistency_time_ms=timing.get("consistency", 0.0),
        )

        restored_reports = [
            {"version_id": versions[i].version_id,
             "transforms": versions[i].transforms_applied,
             "report": reports[i]}
            for i in range(1, len(reports)) if reports[i] is not None
        ]

        aedl_analysis = AEDLAnalysis(
            evasion_score=consistency.evasion_score,
            evasion_level=consistency.evasion_level,
            evasion_type=consistency.evasion_type,
            anomalies_detected=consistency.anomalies_detected,
            trigger_details=consistency.trigger_details,
            evidence_chain=consistency.evidence_chain,
            needs_manual_review=needs_review,
            review_priority=priority,
            processing_metadata=meta,
            restored_reports=restored_reports,
        )

        return AEDLResponse(
            detection=original_detection,
            report=original_report,
            aedl_analysis=aedl_analysis,
        )

    # ─────────────── 融合分数 ───────────────

    def _fuse_scores(self, reports: List[dict | None],
                    triggered_labels: set = None) -> Dict[str, float]:
        """E.1 类别分数融合：只来自原 API 的 category_scores 字段。

        重要设计原则：
        - category_scores 是类别分数的**唯一来源**
        - harmful_segments[].score **不投影到标签**（segment 的 category
          可能是模型误判，投影会放大错误——例如血腥视频被误标为辱骂）
        - calibrated_score **不投影到标签**（它是总体违规分，不对应特定类别）
        - 长短标签名都支持
        - 还原版本的分数只在「触发跃升」时融合，避免变换后构图变化
          导致的模型误判污染最终结果

        参数：
        - triggered_labels: 在一致性校验中触发跃升的标签集合。
          这些标签说明还原后模型看到更多违规内容，取 max。
          未触发的标签保留原始版本的真实分数。
        """
        if triggered_labels is None:
            triggered_labels = set()
        final = {l: 0.0 for l in LABELS}
        if not reports or reports[0] is None:
            return final

        # 先取原始版本的分数作为基底
        orig_detection = reports[0].get("detection") or reports[0]
        orig_scores = orig_detection.get("category_scores") or {}
        for key, val in orig_scores.items():
            short = LONG_LABEL_MAP.get(key, key)
            if short in final:
                final[short] = max(final[short], float(val))

        # 只对触发跃升的标签融合还原版本的高分
        # 未触发的标签保留原始版本的真实分数
        for r in reports[1:]:
            if r is None:
                continue
            detection = r.get("detection") or r
            scores = detection.get("category_scores") or {}
            for key, val in scores.items():
                short = LONG_LABEL_MAP.get(key, key)
                if short in final and short in triggered_labels:
                    final[short] = max(final[short], float(val))
        return final

    @staticmethod
    def _fuse_anomaly_score(reports: List[dict | None],
                             final_scores: Dict[str, float],
                             triggered_labels: set = None) -> float:
        """融合 anomaly_score：用 calibrated_score 兜底。

        原 API 的契约：
        - anomaly_score：总体违规分（可能被推理增强校准过）
        - calibrated_score：推理增强后的校准分数（最准确）
        - category_scores：各类别分数（可能都低于阈值）

        融合策略（与 category_scores 的严格策略不同，这里更宽松）：
        1. 原始版本的 calibrated/anomaly score 始终纳入（保底）
        2. 还原版本的 calibrated/anomaly score 在以下任一条件满足时纳入：
           a. 有触发跃升的标签（triggered_labels 非空）
           b. 还原版本 is_harmful=true（模型判定违规）
           c. 还原版本 calibrated_score > 原始 calibrated_score + 0.1（显著提升）
        3. 取融合后的 category_scores 最大值
        4. 返回以上三者的最大值

        区别于 category_scores 的严格策略：
        - category_scores 只在 flagged 时融合，避免类别污染
        - anomaly_score 更宽松，避免漏检还原版本的真实违规信号
        """
        if triggered_labels is None:
            triggered_labels = set()
        if not reports or reports[0] is None:
            return max(final_scores.values()) if final_scores else 0.0

        # 原始版本的分数作为保底
        orig_detection = reports[0].get("detection") or reports[0]
        orig_calibrated = float(orig_detection.get("calibrated_score", 0.0))
        orig_anomaly = float(orig_detection.get("anomaly_score", 0.0))

        max_calibrated = orig_calibrated
        max_anomaly = orig_anomaly

        # 还原版本：检查是否带来显著的违规信号
        for i, r in enumerate(reports[1:], start=1):
            if r is None:
                continue
            detection = r.get("detection") or r
            restored_calibrated = float(detection.get("calibrated_score", 0.0))
            restored_anomaly = float(detection.get("anomaly_score", 0.0))
            restored_harmful = bool(detection.get("is_harmful", False))

            # 纳入条件：有触发跃升 / 还原版本违规 / 还原分数显著提升
            should_include = (
                triggered_labels  # 有标签触发跃升
                or restored_harmful  # 还原版本判定违规
                or restored_calibrated > orig_calibrated + 0.1  # 显著提升
            )
            if should_include:
                max_calibrated = max(max_calibrated, restored_calibrated)
                max_anomaly = max(max_anomaly, restored_anomaly)

        max_cat = max(final_scores.values()) if final_scores else 0.0
        return max(max_calibrated, max_anomaly, max_cat)

    @staticmethod
    def _get_threshold(detection: dict) -> float:
        """从 detection 中提取阈值（前端可能传入）。默认 0.5。"""
        # 原 API 不返回 threshold，这里用配置默认值
        return 0.5

    # ─────────────── 合并有害时间段 ───────────────

    @staticmethod
    def _merge_harmful_segments(reports: List[dict | None]) -> List[dict]:
        """合并所有版本的有害时间段，按 (start, end, category) 去重。"""
        seen = set()
        merged: List[dict] = []
        for r in reports:
            if r is None:
                continue
            segs = (r.get("detection") or r).get("harmful_segments") or []
            for s in segs:
                key = (round(s.get("start_time", 0), 2),
                       round(s.get("end_time", 0), 2),
                       s.get("category_en", ""))
                if key not in seen:
                    seen.add(key)
                    merged.append(s)
        return merged

    @staticmethod
    def _merge_keyframe_urls(reports: List[dict | None]) -> List[str]:
        """合并所有版本的关键帧 URL，去重保序。"""
        urls: List[str] = []
        seen = set()
        for r in reports:
            if r is None:
                continue
            for u in (r.get("detection") or r).get("keyframe_urls") or []:
                if u not in seen:
                    seen.add(u)
                    urls.append(u)
        return urls

    # ─────────────── 合并有害内容详情 ───────────────

    @staticmethod
    def _merge_harmful_contents(reports: List[dict | None]) -> List[dict]:
        """合并 report.harmful_contents，按 category_en 去重取最大置信度。"""
        best: Dict[str, dict] = {}
        for r in reports:
            if r is None:
                continue
            for hc in (r.get("report") or {}).get("harmful_contents") or []:
                cat = hc.get("category_en", "")
                if cat not in best or hc.get("confidence", 0) > best[cat].get("confidence", 0):
                    best[cat] = hc
        return list(best.values())

    # ─────────────── 预警等级判定 ───────────────

    @staticmethod
    def _alert_level(score: float) -> str:
        """与 API.md §8.2 一致的预警等级判定。"""
        for level, thr in ALERT_THRESHOLDS:
            if score >= thr:
                return level
        return "SAFE"

    # ─────────────── 摘要与处置建议 ───────────────

    @staticmethod
    def _build_summary(original_summary: str, anomaly_score: float,
                       predicted_cats: List[str],
                       consistency: ConsistencyResult) -> str:
        """在原始摘要基础上追加规避信息（若有）。

        前缀 [AEDL-FIX-v3] 用于让用户从输出立即确认服务器加载的是
        修复后的代码（解决了血腥视频被误判为辱骂的问题）。
        如果输出中没有此前缀，说明服务器仍在运行旧代码，需要真正重启。
        """
        cat_zh = "、".join(predicted_cats) if predicted_cats else "无"
        summary = f"[AEDL-FIX-v3] 检测到异常评分 {anomaly_score:.2f}，涉及类别：{cat_zh}"
        if consistency.evasion_level != "none" and consistency.evidence_chain:
            summary += f"；规避分析：{consistency.evidence_chain}"
        return summary

    @staticmethod
    def _build_action_suggestion(alert_level: str,
                                 consistency: ConsistencyResult) -> str:
        """根据预警等级与规避等级生成处置建议。"""
        base = {
            "HIGH": "建议立即下架并转人工审核，等待进一步处理",
            "MEDIUM": "建议限制推荐并人工复审，确认内容性质",
            "LOW": "建议标记待审，纳入审核队列等待处理",
            "SAFE": "内容正常，建议常规监控",
        }.get(alert_level, "建议人工复核")

        if consistency.evasion_level == "high":
            base += "；检测到高风险规避嫌疑，强制人工复核"
        elif consistency.evasion_level == "medium":
            base += "；检测到中风险规避嫌疑，优先人工复核"
        return base
