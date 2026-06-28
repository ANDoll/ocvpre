"""AEDL: 对抗规避检测层 (Adversarial Evasion Detection Layer)

独立中间模块，部署于现有视频审核模型的上游。
接收前端上传视频 → 检测规避手段 → 变换还原 → 调用原 /api/v1/detect → 一致性校验 → 融合输出。
"""

__version__ = "1.0.0"
