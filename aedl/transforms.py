"""模块 C：变换还原层（逆向变换执行）

对应方案文档「四、模块 C：变换还原层」。
对原始视频施加可逆或语义保持变换，生成候选还原版本。
"""
from __future__ import annotations

import os
import time
import uuid
from typing import List, Tuple

import cv2
import numpy as np

from .config import TransformConfig
from .schemas import RoutingDecision, RestoredVersion, TransformSpec


class TransformRestorer:
    """变换还原层：执行策略路由层选定的变换。"""

    def __init__(self, cfg: TransformConfig, temp_dir: str):
        self.cfg = cfg
        self.temp_dir = temp_dir
        os.makedirs(temp_dir, exist_ok=True)

    def build_versions(self, video_path: str, decision: RoutingDecision
                       ) -> Tuple[List[RestoredVersion], float]:
        """生成原始 + 还原版本列表。"""
        t0 = time.time()
        versions: List[RestoredVersion] = [
            RestoredVersion(version_id="original", video_path=video_path, is_original=True)
        ]
        if not decision.triggered:
            return versions, (time.time() - t0) * 1000

        for chain in decision.transform_chains:
            vid = f"restored_{len(versions)}"
            try:
                out_path = self._apply_chain(video_path, chain, vid)
                versions.append(RestoredVersion(
                    version_id=vid,
                    video_path=out_path,
                    transforms_applied=[s.name for s in chain],
                ))
            except Exception as e:
                # 单条变换链失败不影响其他版本
                print(f"[AEDL] transform chain {chain} failed: {e}")
                continue
        return versions, (time.time() - t0) * 1000

    # ─────────────── 变换链执行 ───────────────

    def _apply_chain(self, video_path: str, chain: List[TransformSpec], version_id: str) -> str:
        """按顺序对视频执行变换链，输出新视频文件。"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        # 预计算分屏裁剪框（若链中含 split_extract，输出尺寸会变）
        out_w, out_h = w, h
        crop_rect = None
        split_regions = None
        resample_ratio = 1.0

        for spec in chain:
            if spec.name == "center_crop":
                ratio = spec.params.get("ratio", self.cfg.center_crop_default)
                crop_rect = self._compute_crop_rect(w, h, ratio)
                out_w, out_h = crop_rect[2], crop_rect[3]
            elif spec.name == "split_extract":
                split_regions = self._compute_split_regions(w, h, spec.params.get("lines", []))
                if split_regions:
                    # 取最大子画面
                    r0 = max(split_regions, key=lambda r: r[2] * r[3])
                    out_w, out_h = r0[2], r0[3]
            elif spec.name == "resample":
                resample_ratio = spec.params.get("ratio", 1.0)

        out_path = os.path.join(self.temp_dir, f"{version_id}_{uuid.uuid4().hex[:8]}.mp4")
        # 变速还原时保持原始 fps，通过帧重复/跳帧改变时长，而不是改 fps
        # 这样审核模型抽帧密度不会被打乱
        out_fps = fps
        writer = cv2.VideoWriter(out_path, fourcc, out_fps, (out_w, out_h))

        # 变速还原参数：
        # - 加速（ratio > 1）：用户跳帧让视频变快
        #   还原策略：每帧重复 ratio 次，让视频恢复原始时长
        #   目的：让审核模型有更多抽帧机会，不会错过违规瞬间
        # - 减速（ratio < 1）：用户复制帧让视频变慢
        #   还原策略：跳过部分帧，恢复原始节奏
        repeat_count = int(round(resample_ratio)) if resample_ratio > 1.0 else 1
        skip_period = int(round(1.0 / resample_ratio)) if resample_ratio < 1.0 else 1

        try:
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # 减速还原：按 skip_period 跳过部分帧
                if skip_period > 1 and frame_idx % skip_period != 0:
                    frame_idx += 1
                    continue
                for spec in chain:
                    frame = self._apply_one(frame, spec, crop_rect, split_regions)
                    if frame is None or frame.size == 0:
                        break
                # 加速还原：每帧重复写入 repeat_count 次
                if frame is not None and frame.size > 0:
                    for _ in range(repeat_count):
                        writer.write(frame)
                frame_idx += 1
        finally:
            cap.release()
            writer.release()
        return out_path

    def _apply_one(self, frame: np.ndarray, spec: TransformSpec,
                   crop_rect, split_regions) -> np.ndarray:
        """对单帧执行单个变换。"""
        if spec.name == "mirror":
            return cv2.flip(frame, 1)
        if spec.name == "vflip":
            return cv2.flip(frame, 0)  # 垂直翻转
        if spec.name == "center_crop":
            x, y, w, h = crop_rect
            return frame[y:y + h, x:x + w]
        if spec.name == "brighten":
            return self._brighten(frame)
        if spec.name == "denoise":
            use_g = spec.params.get("use_gaussian", True)
            return self._denoise(frame, use_g)
        if spec.name == "split_extract":
            return self._extract_main_region(frame, split_regions)
        if spec.name == "resample":
            return frame  # 抽帧在 _apply_chain 中处理
        return frame

    # ─────────────── C.1 镜像 ───────────────

    @staticmethod
    def _mirror(frame: np.ndarray) -> np.ndarray:
        return cv2.flip(frame, 1)

    # ─────────────── C.2 中心裁剪 ───────────────

    def _compute_crop_rect(self, w: int, h: int, ratio: float) -> Tuple[int, int, int, int]:
        ratio = max(self.cfg.center_crop_min, min(self.cfg.center_crop_max, ratio))
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        x = (w - new_w) // 2
        y = (h - new_h) // 2
        return x, y, new_w, new_h

    # ─────────────── C.3 亮度增强 ───────────────

    def _brighten(self, frame: np.ndarray) -> np.ndarray:
        """Gamma 校正 + CLAHE。"""
        inv_gamma = 1.0 / self.cfg.gamma_value
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in np.arange(256)]).astype("uint8")
        frame = cv2.LUT(frame, table)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.cfg.clahe_clip_limit, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # ─────────────── C.4 去噪 ───────────────

    def _denoise(self, frame: np.ndarray, use_gaussian: bool) -> np.ndarray:
        if use_gaussian:
            return cv2.GaussianBlur(frame, (self.cfg.gaussian_kernel, self.cfg.gaussian_kernel),
                                    self.cfg.gaussian_sigma)
        return cv2.fastNlMeansDenoisingColored(frame, None, self.cfg.denoise_h, 10, 7, 21)

    # ─────────────── C.5 抽帧密度补偿 ───────────────
    # 见 _apply_chain 中的跳帧逻辑

    # ─────────────── C.6 分屏主画面提取 ───────────────

    def _compute_split_regions(self, w: int, h: int, lines: List[dict]
                               ) -> List[Tuple[int, int, int, int]]:
        """根据分割线计算各子画面区域 (x, y, w, h)。"""
        if not lines:
            return [(0, 0, w, h)]
        v_lines = sorted([ln["position"] for ln in lines if ln["orientation"] == "vertical"])
        h_lines = sorted([ln["position"] for ln in lines if ln["orientation"] == "horizontal"])

        x_cuts = [0.0] + [p * w for p in v_lines] + [float(w)]
        y_cuts = [0.0] + [p * h for p in h_lines] + [float(h)]

        regions = []
        for i in range(len(x_cuts) - 1):
            for j in range(len(y_cuts) - 1):
                rx = int(x_cuts[i])
                ry = int(y_cuts[j])
                rw = int(x_cuts[i + 1] - x_cuts[i])
                rh = int(y_cuts[j + 1] - y_cuts[j])
                if rw > 10 and rh > 10:
                    regions.append((rx, ry, rw, rh))

        # 仅保留面积超过主画面 50% 的子画面
        if regions:
            max_area = max(r[2] * r[3] for r in regions)
            regions = [r for r in regions if r[2] * r[3] >= max_area * self.cfg.split_subarea_min_ratio]
        return regions or [(0, 0, w, h)]

    def _extract_main_region(self, frame: np.ndarray, regions) -> np.ndarray:
        if not regions:
            return frame
        # 取面积最大的
        r0 = max(regions, key=lambda r: r[2] * r[3])
        x, y, w, h = r0
        return frame[y:y + h, x:x + w]
