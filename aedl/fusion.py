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
        is_split_screen = "split_screen" in consistency.anomalies_detected
        final_scores = self._fuse_scores(reports, triggered_labels, is_split_screen)

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
        # is_harmful 用 anomaly_score 判断（基于还原版本信号动态调整）
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
        # 分屏/屏中屏检测：强制人工审核（不依赖原模型判断，因为原模型对缩小画面检测能力有限）
        forced_review = "split_screen" in consistency.anomalies_detected
        needs_review = forced_review or consistency.evasion_level in ("medium", "high")
        if forced_review:
            priority = "high"
        else:
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
                    triggered_labels: set = None,
                    is_split_screen: bool = False) -> Dict[str, float]:
        """E.1 类别分数融合：只来自原 API 的 category_scores 字段。

        重要设计原则：
        - category_scores 是类别分数的**唯一来源**
        - harmful_segments[].score **不投影到标签**（segment 的 category
          可能是模型误判，投影会放大错误——例如血腥视频被误标为辱骂）
        - calibrated_score **不投影到标签**（它是总体违规分，不对应特定类别）
        - 长短标签名都支持

        普通模式：
        - 还原版本的分数只在「触发跃升」时融合，避免变换后构图变化
          导致的模型误判污染最终结果

        屏中屏模式（is_split_screen=True）：
        - 还原视频后比较每个类别还原前后的分数
        - 涨得越多的类别，乘以越大的放大系数（精准放大真实信号）
        - 涨得少或不涨的类别，不放大，保留还原版本的真实分数
        - 这样最终 predicted_categories 精准反映真实涨起来的类别
          （如金钱欺诈视频还原后 Money 涨 380 倍 → Money 被放大 → 判为 Money）
          而不是误判为其他类别（如血腥）

        参数：
        - triggered_labels: 在一致性校验中触发跃升的标签集合
        - is_split_screen: 是否检测到屏中屏。True 时启用按涨跌幅精准放大
        """
        if triggered_labels is None:
            triggered_labels = set()
        final = {l: 0.0 for l in LABELS}
        if not reports or reports[0] is None:
            return final

        # 先取原始版本的分数作为基底
        orig_detection = reports[0].get("detection") or reports[0]
        orig_scores = orig_detection.get("category_scores") or {}
        orig_cat: Dict[str, float] = {}
        for key, val in orig_scores.items():
            short = LONG_LABEL_MAP.get(key, key)
            if short in final:
                final[short] = max(final[short], float(val))
                orig_cat[short] = float(val)

        # 屏中屏模式：按涨跌幅精准放大
        if is_split_screen:
            return self._boost_scores_by_evidence(final, orig_cat, reports)

        # 普通模式：只对触发跃升的标签融合还原版本的高分
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
    def _boost_scores_by_evidence(final: Dict[str, float],
                                   orig_cat: Dict[str, float],
                                   reports: List[dict | None]) -> Dict[str, float]:
        """屏中屏模式：按类别涨跌幅精准放大分数。

        原理：
        - 屏中屏视频还原后，被掩盖的真实类别分数会显著提升
        - 涨得越多 → 系数越大（说明这就是被掩盖的真实类别）
        - 涨得少或不涨 → 不放大（说明不是被掩盖的类别）

        计算方法：
        1. 对每个类别，找到还原版本中的最高分 max_restored
        2. 计算真实涨跌倍数 ratio = max_restored / original（不封顶）
        3. 根据 ratio 决定放大系数（涨得越多 → 系数越大）：
           - ratio > 200 → 系数 25（极强信号，如从 0.0001 涨到 0.025+，380 倍）
           - ratio > 50  → 系数 20
           - ratio > 10  → 系数 10
           - ratio > 3   → 系数 5
           - ratio > 1.5 → 系数 2
           - ratio <= 1.5 → 系数 1（不放大）
        4. final_score = min(max_restored * 系数, 1.0)

        设计依据：
        - 还原版本的真实分数是模型实际输出（不是构造的）
        - 放大系数只是"增强已检测到的信号"，不是无中生有
        - 涨得少不放大，避免误判（如 Blood 涨 1.4 倍 → 不放大，避免误判为血腥）
        """
        # 先收集每个类别在所有还原版本中的最高分
        max_restored: Dict[str, float] = {l: 0.0 for l in LABELS}
        for r in reports[1:]:
            if r is None:
                continue
            detection = r.get("detection") or r
            scores = detection.get("category_scores") or {}
            for key, val in scores.items():
                short = LONG_LABEL_MAP.get(key, key)
                if short in max_restored:
                    max_restored[short] = max(max_restored[short], float(val))

        # 对每个类别按涨跌幅精准放大
        for label in LABELS:
            orig_v = orig_cat.get(label, 0.0)
            restored_v = max_restored.get(label, 0.0)

            # 计算真实涨跌倍数（不封顶，体现真实信号强度）
            if orig_v < 1e-6:
                # 原始为 0：用 restored_v 是否显著判断
                ratio = 1000.0 if restored_v > 0.01 else 1.0
            else:
                ratio = restored_v / orig_v

            # 按倍数决定放大系数
            # 涨得越多 → 系数越大（精准放大真实信号）
            if ratio > 200:
                boost = 25.0
            elif ratio > 50:
                boost = 20.0
            elif ratio > 10:
                boost = 10.0
            elif ratio > 3:
                boost = 5.0
            elif ratio > 1.5:
                boost = 2.0
            else:
                boost = 1.0  # 不放大

            # final = 还原版本最高分 × 系数（封顶 1.0）
            boosted = min(restored_v * boost, 1.0)
            # 至少不低于原始分数（避免还原后变低）
            final[label] = max(final.get(label, 0.0), boosted, orig_v)

        return final

    @staticmethod
    def _fuse_anomaly_score(reports: List[dict | None],
                             final_scores: Dict[str, float],
                             triggered_labels: set = None) -> float:
        """融合 anomaly_score：基于还原版本信号动态调整。

        原 API 的契约：
        - anomaly_score：总体违规分（可能被推理增强校准过）
        - calibrated_score：推理增强后的校准分数（最准确）
        - category_scores：各类别分数（可能都低于阈值）
        - ood_score：out-of-distribution 分数，内容偏离训练分布的程度
          还原后 ood_score 飙升说明模型看到了"不正常"的内容

        融合策略（基于还原版本信号，无信号则不强行拔高）：
        1. 原始版本的 calibrated/anomaly score 始终纳入（保底）
        2. 还原版本在以下任一条件满足时纳入 calibrated/anomaly：
           a. 有触发跃升的标签（triggered_labels 非空）
           b. 还原版本 is_harmful=true（模型判定违规）
           c. 还原版本 calibrated_score > 原始 + 0.05（显著提升）
           d. 类别信号提升：某类别绝对提升 > 0.03 或相对提升 > 3 倍
           e. ood 信号：还原版本 ood_score > 0.3 且 > 原始 + 0.2
        3. 纳入的信号包括：calibrated, anomaly, 类别信号, ood 信号
        4. 返回所有纳入信号的最大值

        区别于一刀切强制 0.5：
        - 无还原版本信号时，保留原始版本真实分数
        - 有信号时，按信号强度动态提升
        """
        if triggered_labels is None:
            triggered_labels = set()
        if not reports or reports[0] is None:
            return max(final_scores.values()) if final_scores else 0.0

        # 原始版本的分数作为保底
        orig_detection = reports[0].get("detection") or reports[0]
        orig_calibrated = float(orig_detection.get("calibrated_score", 0.0))
        orig_anomaly = float(orig_detection.get("anomaly_score", 0.0))
        orig_ood = float(orig_detection.get("ood_score", 0.0))
        # 原始版本的类别分数（归一化长短标签）
        orig_scores_raw = orig_detection.get("category_scores") or {}
        orig_cat: Dict[str, float] = {}
        for key, val in orig_scores_raw.items():
            short = LONG_LABEL_MAP.get(key, key)
            if short in LABELS:
                orig_cat[short] = float(val)

        max_calibrated = orig_calibrated
        max_anomaly = orig_anomaly
        max_cat_signal = 0.0  # 还原版本带来的类别信号
        max_ood_signal = 0.0  # 还原版本带来的 ood 异常信号

        # 还原版本：检查是否带来显著的违规信号
        for i, r in enumerate(reports[1:], start=1):
            if r is None:
                continue
            detection = r.get("detection") or r
            restored_calibrated = float(detection.get("calibrated_score", 0.0))
            restored_anomaly = float(detection.get("anomaly_score", 0.0))
            restored_ood = float(detection.get("ood_score", 0.0))
            restored_harmful = bool(detection.get("is_harmful", False))
            restored_scores_raw = detection.get("category_scores") or {}

            # 检查类别信号提升：
            # - 绝对提升 > 0.03（捕捉弱信号，之前 0.1 太严）
            # - 相对提升 > 3 倍且还原值 > 0.03（捕捉数量级提升）
            cat_signal = False
            for key, val in restored_scores_raw.items():
                short = LONG_LABEL_MAP.get(key, key)
                if short in LABELS:
                    orig_v = orig_cat.get(short, 0.0)
                    restored_v = float(val)
                    delta = restored_v - orig_v
                    ratio = restored_v / max(orig_v, 0.001)
                    if delta > 0.03 or (ratio > 3.0 and restored_v > 0.03):
                        cat_signal = True
                        max_cat_signal = max(max_cat_signal, restored_v)

            # 检查 ood 信号：还原后 ood 飙升说明模型看到异常内容
            if restored_ood > 0.3 and restored_ood > orig_ood + 0.2:
                max_ood_signal = max(max_ood_signal, restored_ood)

            # 纳入条件：有触发跃升 / 还原版本违规 / 还原分数显著提升 / 类别信号 / ood 信号
            should_include = (
                triggered_labels  # 有标签触发跃升
                or restored_harmful  # 还原版本判定违规
                or restored_calibrated > orig_calibrated + 0.05  # 显著提升
                or cat_signal  # 类别信号提升
                or max_ood_signal > 0  # ood 异常信号
            )
            if should_include:
                max_calibrated = max(max_calibrated, restored_calibrated)
                max_anomaly = max(max_anomaly, restored_anomaly)

        max_cat = max(final_scores.values()) if final_scores else 0.0
        return max(max_calibrated, max_anomaly, max_cat, max_cat_signal, max_ood_signal)

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
        # 分屏/屏中屏强制人工审核（原模型对缩小画面检测能力有限）
        if "split_screen" in consistency.anomalies_detected:
            base += "；检测到分屏/屏中屏特征，强制人工审核"
        return base
