"""CLIP re-rank 模块：屏中屏模式下用纯 CLIP 重新计算 category_scores。

原 NSFW Detector 模型在 anomaly_score < 0.5（is_harmful=false）时，
category_scores 基于 SVLA logits2 的条件概率分配，可能有偏置。
当 anomaly_score >= 0.5 时，原模型会用 CLIP re-rank 覆盖 category_scores。

本模块在 AEDL 侧自做 CLIP re-rank：
- 即使还原版本的 anomaly_score < 0.5（未触发原模型 re-rank），
  AEDL 也用纯 CLIP 信号重新计算 category_scores。
- 这样类别分数更准确，避免 SVLA logits2 偏置导致的误报。

复用网页模型的 CLIP 配置（ViT-B/16，7 类，每类 3 prompts）。
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional

import cv2
import numpy as np

from .config import ClipRerankConfig


# 7 类的文本 prompts（与网页模型 feature_extractor.py CLASS_PROMPTS 一致）
CLASS_PROMPTS = {
    "Smoke": [
        "a person smoking a cigarette",
        "someone holding and smoking tobacco",
        "smoke coming from a cigarette or pipe",
    ],
    "Blood": [
        "blood on the ground, bloody scene, gore",
        "a person bleeding from an injury",
        "graphic medical scene with blood",
    ],
    "Violent": [
        "people fighting and hitting each other",
        "physical altercation and violence",
        "riot or street fight scene",
    ],
    "Abusive": [
        "person making aggressive gestures",
        "verbal harassment and threatening behavior",
        "someone using abusive language or gestures",
    ],
    "Sexy": [
        "sexually suggestive content and exposure",
        "inappropriate revealing clothing or acts",
        "explicit adult content",
    ],
    "Money": [
        "displaying large amounts of cash suspiciously",
        "gambling or scam related content",
        "fraudulent money scheme promotion",
    ],
    "Policy": [
        "politically sensitive content and symbols",
        "unauthorized political commentary",
        "political slander or misinformation",
    ],
}

LABELS = ["Smoke", "Blood", "Violent", "Abusive", "Sexy", "Money", "Policy"]


class ClipReranker:
    """CLIP re-rank 单例：加载 CLIP 模型，对视频重新计算类别分数。

    单例模式：CLIP 模型加载较慢，整个 AEDL 生命周期只加载一次。
    """

    _instance: Optional["ClipReranker"] = None
    _initialized: bool = False

    def __new__(cls, cfg: ClipRerankConfig = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, cfg: ClipRerankConfig = None):
        if self._initialized:
            return
        if cfg is None:
            cfg = ClipRerankConfig()
        self.cfg = cfg
        self._load_clip()
        self._init_text_features()
        ClipReranker._initialized = True

    def _load_clip(self):
        """加载 CLIP 模型（复用网页模型的 clip 包）。

        网页模型的 clip 包用了相对导入（from .model import build_model），
        必须把其父目录加到 sys.path，然后用 `import clip` 作为包导入。
        """
        clip_path = self.cfg.clip_module_path
        if clip_path and os.path.isdir(clip_path):
            # clip_module_path 指向 NFSW_Detector/clip 目录
            # 需要把其父目录（NFSW_Detector）加到 sys.path
            parent_dir = os.path.dirname(clip_path)
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)

        import torch
        import clip

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(self.cfg.clip_variant, device=self.device)
        self.model.eval()
        if self.device.type == "cuda":
            self.model = self.model.float()
        self.torch = torch  # 保存引用，避免后续重复 import

    def _init_text_features(self):
        """预计算 7 类的文本特征（3-prompt 平均）。"""
        text_feats_all = {}
        for cat_en, prompts in CLASS_PROMPTS.items():
            text_feats = self._extract_text_features(prompts)
            mean_feat = text_feats.mean(axis=0)
            mean_feat = mean_feat / (np.linalg.norm(mean_feat) + 1e-10)
            text_feats_all[cat_en] = mean_feat
        self.text_feats = text_feats_all

    def _extract_text_features(self, text_prompts: List[str]) -> np.ndarray:
        """提取文本特征，复用网页模型 CLIPFeatureExtractor 的逻辑。"""
        clip = sys.modules.get("clip")
        tokens = clip.tokenize(text_prompts).to(self.device)
        with self.torch.no_grad():
            token_embedding = self.model.encode_token(tokens)
            text_features = self.model.encode_text(token_embedding, tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy()

    def _extract_visual_features(self, frames: np.ndarray) -> np.ndarray:
        """提取视觉特征，复用网页模型 CLIPFeatureExtractor 的逻辑。"""
        from PIL import Image

        all_features = []
        num_frames = frames.shape[0]
        batch_size = self.cfg.batch_size
        for start in range(0, num_frames, batch_size):
            end = min(start + batch_size, num_frames)
            batch_frames = frames[start:end]
            batch_tensors = []
            for frame in batch_frames:
                image = Image.fromarray(frame)
                tensor = self.preprocess(image)
                batch_tensors.append(tensor)
            batch_tensor = self.torch.stack(batch_tensors).to(self.device)
            with self.torch.no_grad():
                batch_features = self.model.encode_image(batch_tensor)
                batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
            all_features.append(batch_features.cpu().numpy())
        return np.concatenate(all_features, axis=0)

    def _sample_frames(self, video_path: str) -> np.ndarray:
        """从视频均匀采样帧，返回 [N, H, W, 3] RGB 数组。"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return np.zeros((0, 224, 224, 3), dtype=np.uint8)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        num_samples = min(self.cfg.num_frames, total_frames)
        if num_samples <= 0:
            cap.release()
            return np.zeros((0, 224, 224, 3), dtype=np.uint8)

        # 均匀采样
        indices = np.linspace(0, max(0, total_frames - 1), num_samples).astype(int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                # BGR → RGB
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        if not frames:
            return np.zeros((0, 224, 224, 3), dtype=np.uint8)
        return np.array(frames)

    def rerank(self, video_path: str, anomaly_score: float) -> Dict[str, float]:
        """对视频做 CLIP re-rank，返回新的 category_scores。

        复用网页模型的 re-rank 逻辑：
        - 抽帧 → CLIP 视觉特征
        - 计算每帧与每个类别文本特征的相似度
        - 取 top-k 帧相似度平均
        - category_scores[cat] = anomaly_score × max(0, topk_sim)

        Args:
            video_path: 视频文件路径
            anomaly_score: 该版本的原 anomaly_score（作为加权基数）

        Returns:
            新的 category_scores dict，key 为 7 类短标签
        """
        frames = self._sample_frames(video_path)
        if len(frames) == 0:
            return {l: 0.0 for l in LABELS}

        # 提取视觉特征
        visual_feats = self._extract_visual_features(frames)
        visual_feats_norm = visual_feats / (
            np.linalg.norm(visual_feats, axis=-1, keepdims=True) + 1e-10
        )

        # 计算每个类别的 top-k 相似度平均
        k = min(self.cfg.topk_frames, len(visual_feats_norm))
        rerank_scores = {}
        for cat_en, text_feat in self.text_feats.items():
            sim = visual_feats_norm @ text_feat  # [N]
            topk_sim = float(np.sort(sim)[-k:].mean())
            rerank_scores[cat_en] = anomaly_score * max(0.0, topk_sim)

        return rerank_scores
