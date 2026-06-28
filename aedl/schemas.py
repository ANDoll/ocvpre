"""AEDL 数据模型：模块间传递的标准结构。

对应方案文档「四、核心模块详细设计」中各模块的输入输出。
"""
from __future__ import annotations

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field


# ───────────────────────── 模块 A 输出 ─────────────────────────

class SplitLine(BaseModel):
    position: float          # 直线位置（像素或比例）
    orientation: Literal["horizontal", "vertical"]


class PerceptionResult(BaseModel):
    """输入感知层输出"""
    is_mirror_suspicious: bool = False
    symmetry_score: float = 0.0
    is_border_occluded: bool = False
    border_width_ratio: float = 0.0
    occlusion_confidence: float = 0.0
    is_quality_degraded: bool = False
    laplacian_score: float = 0.0
    blocking_artifact_score: float = 0.0
    is_split_screen: bool = False
    split_lines: List[SplitLine] = Field(default_factory=list)
    sub_region_count: int = 0
    is_speed_abnormal: bool = False
    estimated_speed_ratio: float = 1.0
    detection_channel: Literal["audio", "visual", "both", "none"] = "none"
    # 扩展：垂直翻转（上下翻转）
    is_vertical_flip_suspicious: bool = False
    vertical_symmetry_score: float = 0.0
    # 扩展：亮度异常（蒙版遮挡/调暗/半透明）
    is_darkened: bool = False
    mean_brightness: float = 0.0
    # 扩展：闪烁规避（插入几帧有害内容）
    is_flickering: bool = False
    flicker_score: float = 0.0
    has_anomaly: bool = False
    anomaly_count: int = 0
    check_time_ms: float = 0.0


# ───────────────────────── 模块 B 输出 ─────────────────────────

class TransformSpec(BaseModel):
    """单次变换规格"""
    name: str                          # mirror / center_crop / brighten / denoise / resample / split_extract
    params: Dict[str, Any] = Field(default_factory=dict)


class RoutingDecision(BaseModel):
    """策略路由层输出"""
    triggered: bool = False
    transform_chains: List[List[TransformSpec]] = Field(default_factory=list)
    # 每条链是一组串联变换，对应一个还原版本
    bypass_reason: Optional[str] = None   # 命中白名单或置信度低时的旁路原因


# ───────────────────────── 模块 C 输出 ─────────────────────────

class RestoredVersion(BaseModel):
    """单个还原版本"""
    version_id: str                    # 原始为 "original"，还原为 "restored_1" 等
    video_path: str
    transforms_applied: List[str] = Field(default_factory=list)
    is_original: bool = False


# ───────────────────────── 模块 D 输出 ─────────────────────────

class TriggerDetail(BaseModel):
    transform: str
    label: str
    original_score: float
    restored_score: float
    delta: float
    flagged: bool


class ConsistencyResult(BaseModel):
    """一致性校验层输出（模块内部使用，不直接对外）"""
    evasion_score: float = 0.0
    evasion_level: Literal["none", "low", "medium", "high"] = "none"
    evasion_type: Optional[str] = None
    anomalies_detected: List[str] = Field(default_factory=list)
    trigger_details: List[TriggerDetail] = Field(default_factory=list)
    evidence_chain: str = ""
    label_jumps: Dict[str, float] = Field(default_factory=dict)
    jsd_values: List[float] = Field(default_factory=list)
    max_jsd: float = 0.0


# ───────────────────────── 模块 E 输出（最终响应） ─────────────────────────
#
# 设计原则（方案 A）：
# AEDLResponse 顶层保持原 NSFW Detector 的 {detection, report} 结构不变，
# 前端原有渲染逻辑（data.detection.xxx / data.report.xxx）零改动。
# 规避分析作为可选字段 aedl_analysis 挂载，前端可渐进式接入。

class ProcessingMetadata(BaseModel):
    """AEDL 处理耗时元数据"""
    versions_generated: int
    model_inference_count: int
    total_processing_time_ms: float
    perception_time_ms: float = 0.0
    transform_time_ms: float = 0.0
    inference_time_ms: float = 0.0
    consistency_time_ms: float = 0.0


class AEDLAnalysis(BaseModel):
    """AEDL 规避分析附加字段。

    前端可选择性渲染该字段以展示规避检测详情。
    不渲染该字段不影响原有 detection/report 的展示。
    """
    evasion_score: float = 0.0
    evasion_level: Literal["none", "low", "medium", "high"] = "none"
    evasion_type: Optional[str] = None
    anomalies_detected: List[str] = Field(default_factory=list)
    trigger_details: List[TriggerDetail] = Field(default_factory=list)
    evidence_chain: str = ""
    needs_manual_review: bool = False
    review_priority: Literal["none", "low", "medium", "high"] = "none"
    processing_metadata: Optional[ProcessingMetadata] = None
    # 各还原版本的检测摘要（供前端可选展示多版本对比）
    restored_reports: List[Dict[str, Any]] = Field(default_factory=list)


class AEDLResponse(BaseModel):
    """AEDL 对外最终响应。

    顶层保持原 /api/v1/detect 的 {detection, report} 结构契约不变。
    前端原有渲染代码无需任何修改即可正常工作。

    新增字段：
    - aedl_analysis: 规避分析详情（可选渲染，不影响原有功能）
    """
    # ── 原 API 契约字段（顶层不变，前端零改动） ──
    detection: Optional[Dict[str, Any]] = None
    report: Optional[Dict[str, Any]] = None
    # ── AEDL 附加字段（可选渲染） ──
    aedl_analysis: Optional[AEDLAnalysis] = None


class AEDLError(BaseModel):
    error: str
    detail: str


# ───────────────────────── 测试预览响应 ─────────────────────────
#
# 专门用于独立测试 AEDL 还原功能：输入视频 → 感知 + 变换 → 返回还原视频下载 URL
# 不调用后端 NSFW Detector 模型，方便快速验证还原效果。

class PreviewVersion(BaseModel):
    """还原版本信息（含下载 URL）"""
    version_id: str
    is_original: bool
    transforms_applied: List[str] = Field(default_factory=list)
    video_url: str                       # 可下载的 URL
    video_filename: str


class PreviewResponse(BaseModel):
    """AEDL 还原预览响应。

    用于测试验证：输入一个视频（如镜像翻转的视频），
    返回原始 + 各还原版本的下载 URL，可直接下载查看还原效果。
    """
    perception: PerceptionResult
    routing_triggered: bool
    bypass_reason: Optional[str] = None
    versions: List[PreviewVersion]
    # 测试建议（提示用户该如何验证）
    test_hint: str
