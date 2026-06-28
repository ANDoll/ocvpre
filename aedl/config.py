"""AEDL 配置：所有阈值集中管理，支持外部 YAML 覆盖。

对应方案文档「六、关键参数与阈值汇总」。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

import yaml


@dataclass
class PerceptionConfig:
    """模块 A 阈值"""
    mirror_ssim_threshold: float = 0.95        # 高于此值标记镜像异常
    mirror_saliency_weight: float = 0.5        # 天然对称显著区域降权
    border_ratio_range: tuple = (0.05, 0.10)  # 边框检测像素带范围
    border_variance_threshold: float = 50.0   # 边框方差阈值（低则疑似遮挡）
    laplacian_threshold_720p: float = 80.0
    laplacian_threshold_1080p: float = 150.0
    laplacian_threshold_4k: float = 400.0
    blocking_artifact_threshold: float = 0.5  # 压缩块效应阈值
    split_line_min_ratio: float = 0.8         # 分屏直线最短长度比例
    speed_audio_threshold: float = 1.5        # 音频 1.5 倍速异常
    speed_visual_flow_std: float = 2.0         # 光流偏离 σ 倍数
    max_frames_to_check: int = 3              # 仅检测前 3 个非黑帧
    fast_check_time_ms: int = 50              # 输入感知层总耗时上限
    # 扩展：垂直翻转（上下翻转）检测
    vflip_ssim_threshold: float = 0.90        # 垂直对称性阈值
    # 扩展：亮度异常检测（蒙版/调暗/半透明）
    dark_brightness_threshold: float = 40.0   # 平均亮度低于此值判定为异常暗
    # 扩展：闪烁检测（插入几帧有害内容）
    flicker_check_frames: int = 12            # 闪烁检测采样帧数
    flicker_brightness_threshold: float = 25.0  # 帧间亮度突变阈值


@dataclass
class RouterConfig:
    """模块 B 阈值"""
    max_restored_versions: int = 8            # 含原始共最多 9 版本（支持分屏多组合变换）
    max_transform_chain: int = 2              # 单变换链最多串联 2 项
    min_confidence_to_trigger: float = 0.3    # 低于此值不触发变换
    whitelist_enabled: bool = True
    whitelist_path: str = "configs/whitelist.txt"
    # 保险还原变换：无视感知层是否检测到异常，
    # 默认对所有视频追加这些变换链送审。
    # 由模块 D 一致性校验层通过分数对比判断是否真的存在规避。
    # 支持：mirror（水平翻转）、vflip（垂直翻转）、brighten（亮度增强）
    force_restore_transforms: List[str] = field(default_factory=lambda: ["mirror", "vflip", "brighten"])


@dataclass
class TransformConfig:
    """模块 C 参数"""
    center_crop_min: float = 0.80             # 中心裁剪下限
    center_crop_max: float = 0.95             # 中心裁剪上限
    center_crop_default: float = 0.90
    gamma_value: float = 0.8                  # 亮度 Gamma 校正
    clahe_clip_limit: float = 2.0
    gaussian_kernel: int = 3
    gaussian_sigma: float = 0.8
    denoise_h: float = 3.0                    # fastNlMeans h 参数
    split_subarea_min_ratio: float = 0.5       # 子画面面积超过主画面 50% 才独立提取


@dataclass
class ConsistencyConfig:
    """模块 D 阈值"""
    label_jump_threshold: float = 0.15        # 单标签跃升阈值（降低让还原提升更容易触发）
    label_cliff_threshold: float = 0.3         # 断崖式跃升阈值
    restored_high_confidence: float = 0.5     # 还原后高置信触发线（降低让更多还原版本纳入）
    jsd_threshold: float = 0.15                # JSD 分布偏移阈值
    evasion_weights: tuple = (0.35, 0.25, 0.25, 0.15)  # w1,w2,w3,w4
    high_risk_threshold: float = 0.7
    medium_risk_threshold: float = 0.5
    low_risk_threshold: float = 0.2


@dataclass
class BackendConfig:
    """后端原有 API 配置"""
    base_url: str = "http://localhost:8000"
    detect_endpoint: str = "/api/v1/detect"
    timeout: int = 120                        # 单次推理超时（秒）
    max_retries: int = 2


@dataclass
class ServerConfig:
    """AEDL 服务自身配置"""
    host: str = "0.0.0.0"
    port: int = 8001
    cors_allow_origins: List[str] = field(default_factory=lambda: ["*"])
    temp_dir: str = "temp_aedl"
    max_upload_size_mb: int = 500


@dataclass
class AEDLConfig:
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    transform: TransformConfig = field(default_factory=TransformConfig)
    consistency: ConsistencyConfig = field(default_factory=ConsistencyConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    server: ServerConfig = field(default_factory=ServerConfig)


def load_config(yaml_path: str | None = None) -> AEDLConfig:
    """加载配置，YAML 优先，环境变量次之，最后用默认值。"""
    cfg = AEDLConfig()

    path = yaml_path or os.environ.get("AEDL_CONFIG", "configs/aedl.yaml")
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        _apply_overrides(cfg, data)

    # 环境变量覆盖关键项
    if url := os.environ.get("AEDL_BACKEND_URL"):
        cfg.backend.base_url = url
    if port := os.environ.get("AEDL_PORT"):
        cfg.server.port = int(port)
    return cfg


def _apply_overrides(cfg: AEDLConfig, data: dict) -> None:
    """递归地将 dict 覆盖到 dataclass 字段。"""
    for section_name, section_cfg in cfg.__dict__.items():
        if section_name not in data:
            continue
        section_data = data[section_name]
        if not isinstance(section_data, dict):
            continue
        for k, v in section_data.items():
            if hasattr(section_cfg, k):
                setattr(section_cfg, k, v)
