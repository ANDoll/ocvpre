"""模块 A：输入感知层（快速初筛）

对应方案文档「四、模块 A：输入感知层」。
所有检测在首 3 个非黑帧上完成，毫秒级 CPU 算子。
"""
from __future__ import annotations

import time
from typing import Tuple, List

import cv2
import numpy as np

from .config import PerceptionConfig
from .schemas import PerceptionResult, SplitLine


class InputPerception:
    """输入感知层：对视频首帧进行多维度异常检测。"""

    def __init__(self, cfg: PerceptionConfig):
        self.cfg = cfg

    def analyze(self, video_path: str) -> PerceptionResult:
        t0 = time.time()
        frames, fps, w, h, has_audio = self._sample_frames(video_path)
        if not frames:
            return PerceptionResult(check_time_ms=(time.time() - t0) * 1000)

        mirror_flag, sym_score = self._detect_mirror(frames)
        vflip_flag, vsym_score = self._detect_vflip(frames)
        border_flag, bw_ratio, occ_conf = self._detect_border(frames, w, h)
        degraded_flag, lap_score, block_score = self._detect_quality(frames, w, h)
        split_flag, lines, sub_cnt = self._detect_split_screen(frames, w, h)
        speed_flag, speed_ratio, channel = self._detect_speed(
            video_path, frames, fps, has_audio
        )
        dark_flag, mean_bright = self._detect_darkness(frames)
        flicker_flag, flicker_score = self._detect_flicker(video_path, fps)

        # 屏中屏保险：flicker 触发但 split 未触发时，用帧差法重试
        # 屏中屏内的动态内容会触发闪烁误判，此时应检测到屏中屏
        if flicker_flag and not split_flag:
            inset_lines = self._find_inset_by_frame_diff(video_path, fps, w, h)
            if inset_lines:
                split_flag = True
                lines = inset_lines
                sub_cnt = 2

        anomalies = [mirror_flag, vflip_flag, border_flag, degraded_flag,
                     split_flag, speed_flag, dark_flag, flicker_flag]
        count = sum(anomalies)
        t1 = time.time()

        return PerceptionResult(
            is_mirror_suspicious=mirror_flag,
            symmetry_score=float(sym_score),
            is_vertical_flip_suspicious=vflip_flag,
            vertical_symmetry_score=float(vsym_score),
            is_border_occluded=border_flag,
            border_width_ratio=float(bw_ratio),
            occlusion_confidence=float(occ_conf),
            is_quality_degraded=degraded_flag,
            laplacian_score=float(lap_score),
            blocking_artifact_score=float(block_score),
            is_split_screen=split_flag,
            split_lines=lines,
            sub_region_count=sub_cnt,
            is_speed_abnormal=speed_flag,
            estimated_speed_ratio=float(speed_ratio),
            detection_channel=channel,
            is_darkened=dark_flag,
            mean_brightness=float(mean_bright),
            is_flickering=flicker_flag,
            flicker_score=float(flicker_score),
            has_anomaly=count > 0,
            anomaly_count=count,
            check_time_ms=(t1 - t0) * 1000,
        )

    # ─────────────── 抽帧 ───────────────

    def _sample_frames(self, video_path: str) -> Tuple[List[np.ndarray], float, int, int, bool]:
        """抽取前 3 个非黑帧。返回 (frames, fps, w, h, has_audio)。"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return [], 0.0, 0, 0, False
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        has_audio = int(cap.get(cv2.CAP_PROP_AUDIO_STREAM)) >= 0 if hasattr(cv2, "CAP_PROP_AUDIO_STREAM") else False

        frames: List[np.ndarray] = []
        while len(frames) < self.cfg.max_frames_to_check:
            ret, frame = cap.read()
            if not ret:
                break
            if self._is_black_frame(frame):
                continue
            frames.append(frame)
        cap.release()
        return frames, fps, w, h, has_audio

    @staticmethod
    def _is_black_frame(frame: np.ndarray, threshold: float = 5.0) -> bool:
        """判断是否为黑帧（均值低于 threshold）。"""
        return float(frame.mean()) < threshold

    # ─────────────── A.1 镜像对称性 ───────────────

    def _detect_mirror(self, frames: List[np.ndarray]) -> Tuple[bool, float]:
        """水平镜像 SSIM 对称性检测。"""
        if not frames:
            return False, 0.0
        # 取首帧灰度，缩放加速
        gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (min(480, gray.shape[1]), min(270, gray.shape[0])))
        flipped = cv2.flip(gray, 1)
        sym = self._ssim(gray, flipped)
        flag = sym > self.cfg.mirror_ssim_threshold
        return flag, float(sym)

    def _detect_vflip(self, frames: List[np.ndarray]) -> Tuple[bool, float]:
        """垂直翻转（上下翻转）SSIM 对称性检测。

        与水平镜像同理：对首帧做垂直翻转后计算 SSIM 对称性。
        阈值略低（0.90），因为上下对称的内容比左右对称更少见。
        """
        if not frames:
            return False, 0.0
        gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (min(480, gray.shape[1]), min(270, gray.shape[0])))
        flipped = cv2.flip(gray, 0)  # 垂直翻转
        sym = self._ssim(gray, flipped)
        flag = sym > self.cfg.vflip_ssim_threshold
        return flag, float(sym)

    def _detect_darkness(self, frames: List[np.ndarray]) -> Tuple[bool, float]:
        """亮度异常检测：蒙版遮挡/调暗/半透明导致整体亮度过低。

        当画面平均亮度显著低于正常水平时，可能被深色蒙版覆盖或整体调暗，
        导致审核模型看不清违规内容。
        """
        if not frames:
            return False, 0.0
        brightnesses = [float(f.mean()) for f in frames]
        mean_bright = float(np.mean(brightnesses))
        flag = mean_bright < self.cfg.dark_brightness_threshold
        return flag, mean_bright

    def _detect_flicker(self, video_path: str, fps: float) -> Tuple[bool, float]:
        """闪烁检测：插入几帧有害内容的规避手段。

        特征：大部分帧无害，少数帧有害 → 帧间亮度/内容突变。
        均匀采样若干帧，计算相邻帧亮度差，如果存在突变则判定为闪烁。
        """
        if fps <= 0:
            return False, 0.0
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, 0.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 4:
            cap.release()
            return False, 0.0

        n = min(self.cfg.flicker_check_frames, total)
        # 均匀采样
        indices = np.linspace(0, total - 1, n, dtype=int)
        brightnesses = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                brightnesses.append(float(frame.mean()))
        cap.release()

        if len(brightnesses) < 4:
            return False, 0.0

        # 计算相邻帧亮度差
        diffs = [abs(brightnesses[i + 1] - brightnesses[i]) for i in range(len(brightnesses) - 1)]
        max_diff = max(diffs)
        mean_diff = float(np.mean(diffs))
        std_diff = float(np.std(diffs))

        # 闪烁特征：存在突变（max_diff 远大于 mean_diff）
        # 且突变超过阈值
        flicker_score = max_diff / max(mean_diff + 1e-6, 1.0)
        flag = (max_diff > self.cfg.flicker_brightness_threshold
                and flicker_score > 3.0)
        return flag, float(flicker_score)

    @staticmethod
    def _ssim(a: np.ndarray, b: np.ndarray) -> float:
        """简化 SSIM：用归一化互相关近似，值域 [0,1]。"""
        a = a.astype(np.float32)
        b = b.astype(np.float32)
        c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
        mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
        mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
        mu_a2, mu_b2, mu_ab = mu_a ** 2, mu_b ** 2, mu_a * mu_b
        sig_a2 = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a2
        sig_b2 = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b2
        sig_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_ab
        ssim_map = ((2 * mu_ab + c1) * (2 * sig_ab + c2)) / (
            (mu_a2 + mu_b2 + c1) * (sig_a2 + sig_b2 + c2)
        )
        return float(ssim_map.mean())

    # ─────────────── A.2 边缘遮挡 ───────────────

    def _detect_border(self, frames: List[np.ndarray], w: int, h: int) -> Tuple[bool, float, float]:
        """边缘均匀性检测：判断是否被遮挡。

        分别检测四条边（上/下/左/右），只要任一条边的方差低于阈值且
        内部区域方差显著高于边框，就判定为遮挡。这能正确识别
        「仅上下加黑边」「仅左右加贴纸」等局部遮挡场景。
        """
        if not frames or w == 0 or h == 0:
            return False, 0.0, 0.0
        frame = frames[0]
        bw_min, bw_max = self.cfg.border_ratio_range
        bw_px = int(min(w, h) * bw_max)
        if bw_px < 2:
            return False, 0.0, 0.0

        # 分别计算四条边框像素带的方差
        top = frame[:bw_px, :]
        bottom = frame[-bw_px:, :]
        left = frame[:, :bw_px]
        right = frame[:, -bw_px:]
        top_var = float(top.var())
        bottom_var = float(bottom.var())
        left_var = float(left.var())
        right_var = float(right.var())

        # 内部区域方差
        inner = frame[bw_px:-bw_px, bw_px:-bw_px] if bw_px * 2 < min(w, h) else frame
        inner_var = float(inner.var())

        # 任一条边方差低 + 内部方差显著高 → 遮挡
        threshold = self.cfg.border_variance_threshold
        occluded_edges = [
            ("top", top_var, h),
            ("bottom", bottom_var, h),
            ("left", left_var, w),
            ("right", right_var, w),
        ]
        occluded = [e for e in occluded_edges if e[1] < threshold and inner_var > e[1] * 2]

        if not occluded:
            return False, 0.0, 0.0

        # 取遮挡最严重的那条边
        worst = min(occluded, key=lambda e: e[1])
        # bw_ratio 用对应方向（上下用 h，左右用 w）
        bw_ratio = bw_px / float(worst[2])
        occ_conf = min(1.0, max(0.0, (inner_var - worst[1]) / max(inner_var, 1e-6)))
        return True, float(bw_ratio), float(occ_conf)

    # ─────────────── A.3 画质降质 ───────────────

    def _detect_quality(self, frames: List[np.ndarray], w: int, h: int) -> Tuple[bool, float, float]:
        """拉普拉斯方差 + 压缩块效应。"""
        if not frames:
            return False, 0.0, 0.0
        laps = []
        blocks = []
        for f in frames:
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            laps.append(cv2.Laplacian(gray, cv2.CV_64F).var())
            blocks.append(self._blocking_artifact(gray))
        lap = float(np.mean(laps))
        block = float(np.mean(blocks))

        thr = self._laplacian_threshold(w, h)
        degraded = lap < thr and block > self.cfg.blocking_artifact_threshold
        return degraded, lap, block

    def _laplacian_threshold(self, w: int, h: int) -> float:
        """按分辨率分档。"""
        min_dim = min(w, h)
        if min_dim >= 2000:
            return self.cfg.laplacian_threshold_4k
        if min_dim >= 1000:
            return self.cfg.laplacian_threshold_1080p
        return self.cfg.laplacian_threshold_720p

    @staticmethod
    def _blocking_artifact(gray: np.ndarray) -> float:
        """压缩块效应：8x8 块边界的不连续性强度。

        比较每个 8k 边界两侧像素 (位置 8k-1 与 8k) 的差值。
        块效应严重时，边界两侧出现明显跳变。
        """
        h, w = gray.shape
        if h < 16 or w < 16:
            return 0.0
        block = 8
        # 水平方向块边界：比较位置 (8k-1) 与 (8k)
        h_starts = np.arange(block, w, block)
        if len(h_starts) == 0:
            diff_h_mean = 0.0
        else:
            diff_h = np.abs(
                gray[:, h_starts].astype(np.int16)
                - gray[:, h_starts - 1].astype(np.int16)
            )
            diff_h_mean = float(diff_h.mean())
        # 垂直方向块边界：比较位置 (8k-1) 与 (8k)
        v_starts = np.arange(block, h, block)
        if len(v_starts) == 0:
            diff_v_mean = 0.0
        else:
            diff_v = np.abs(
                gray[v_starts, :].astype(np.int16)
                - gray[v_starts - 1, :].astype(np.int16)
            )
            diff_v_mean = float(diff_v.mean())
        return float((diff_h_mean + diff_v_mean) / 2.0 / 255.0)

    # ─────────────── A.4 分屏 ───────────────

    def _detect_split_screen(self, frames: List[np.ndarray], w: int, h: int) -> Tuple[bool, List[SplitLine], int]:
        """Canny + Hough 直线检测分屏，并验证两侧内容差异。

        分屏的本质特征：长直线 + 两侧区域颜色/亮度分布显著不同。
        仅检测到直线不判定为分屏，避免渐变背景/文字边缘误报。
        """
        if not frames or w == 0:
            return False, [], 0
        gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=80,
            minLineLength=int(min(w, h) * self.cfg.split_line_min_ratio),
            maxLineGap=20,
        )
        candidate_lines: List[SplitLine] = []
        if lines is not None:
            for ln in lines[:, 0]:
                x1, y1, x2, y2 = ln
                length = np.hypot(x2 - x1, y2 - y1)
                if length < min(w, h) * self.cfg.split_line_min_ratio:
                    continue
                if abs(x1 - x2) < 2:           # 近似垂直
                    candidate_lines.append(SplitLine(position=float(x1) / w, orientation="vertical"))
                elif abs(y1 - y2) < 2:         # 近似水平
                    candidate_lines.append(SplitLine(position=float(y1) / h, orientation="horizontal"))
            candidate_lines = self._dedupe_lines(candidate_lines)

        # 验证：分割线两侧区域必须有显著内容差异（颜色直方图距离）
        confirmed: List[SplitLine] = []
        for ln in candidate_lines:
            if self._regions_differ(gray, ln, w, h):
                confirmed.append(ln)
        is_split = len(confirmed) > 0
        sub_count = len(confirmed) + 1 if is_split else 1

        # 备选：屏中屏检测（直线检测失败时，用网格差异检测小窗口）
        if not is_split:
            inset_lines = self._detect_inset_screen(frames[0], w, h)
            if inset_lines:
                return True, inset_lines, 2

        return is_split, confirmed, sub_count

    def _detect_inset_screen(self, frame: np.ndarray, w: int, h: int) -> List[SplitLine]:
        """屏中屏检测：检测画面中小窗口（缩小有害画面嵌入无害背景）。

        方法 1（块方差分析）：
        将画面分成 NxN 网格，计算每块的标准差。
        屏中屏特征：中间区域高方差（视频内容），四周低方差（均匀背景）。
        找到高方差区域的 bounding box 作为屏中屏位置。

        方法 2（轮廓检测，作为备选）：
        Canny + findContours 找矩形轮廓，验证内部/外部内容差异。
        适用于有清晰边框的屏中屏。
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 方法 1：块方差分析（对软边缘屏中屏最有效）
        inset = self._find_inset_by_variance(gray, w, h)
        if inset:
            return inset

        # 方法 2：轮廓检测（对清晰边框屏中屏有效）
        inset = self._find_inset_by_contour(gray, w, h)
        if inset:
            return inset

        return []

    def _find_inset_by_variance(self, gray: np.ndarray, w: int, h: int) -> List[SplitLine]:
        """块方差分析法检测屏中屏。"""
        # 12x12 网格，更精细
        gh, gw = 12, 12
        bh, bw = h // gh, w // gw
        if bh < 10 or bw < 10:
            return []

        # 计算每块的标准差
        std_map = np.zeros((gh, gw), dtype=np.float32)
        for i in range(gh):
            for j in range(gw):
                block = gray[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
                std_map[i, j] = float(block.std())

        # 自适应阈值：整体方差的 0.4 倍，最低 6.0（放宽，避免漏检小窗口）
        global_std = float(gray.std())
        threshold = max(global_std * 0.4, 6.0)

        # 二值化：高方差块为 1（有内容），低方差块为 0（背景）
        binary = (std_map > threshold).astype(np.uint8)

        # 形态学开运算去除孤立噪声
        # 用 2x2 核做开运算（比 3x3 更宽松，保留小窗口）
        kernel = np.ones((2, 2), dtype=np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # 再次膨胀，连接近邻的高方差块
        binary = cv2.dilate(binary, kernel, iterations=1)

        # 找最大连通组件（避免多个小区域干扰）
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if num_labels <= 1:
            return []

        # 找面积最大的非背景组件（label 0 是背景）
        max_area = 0
        max_label = 0
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area > max_area:
                max_area = area
                max_label = label

        if max_label == 0 or max_area < 4:  # 至少 2x2 = 4 块（放宽，支持小窗口）
            return []

        # 只保留最大连通组件
        binary = (labels == max_label).astype(np.uint8)

        # 找 bounding box
        rows = np.any(binary, axis=1)
        cols = np.any(binary, axis=0)
        r_min, r_max = np.where(rows)[0][[0, -1]]
        c_min, c_max = np.where(cols)[0][[0, -1]]

        # 转换为像素坐标
        x1 = int(c_min * bw)
        y1 = int(r_min * bh)
        x2 = int((c_max + 1) * bw)
        y2 = int((r_max + 1) * bh)
        cw = x2 - x1
        ch = y2 - y1

        # 面积筛选：3%-70% 画面（放宽下限，支持小窗口）
        area_ratio = (cw * ch) / (w * h)
        if area_ratio < 0.03 or area_ratio > 0.7:
            return []

        # 内部方差必须显著高于边缘带（放宽条件：1.5 倍而非 2.0）
        margin_blocks = 2
        top_bg = std_map[:margin_blocks, :].mean()
        bottom_bg = std_map[-margin_blocks:, :].mean()
        left_bg = std_map[:, :margin_blocks].mean()
        right_bg = std_map[:, -margin_blocks:].mean()
        inner_std = std_map[r_min:r_max+1, c_min:c_max+1].mean()
        bg_min = min(top_bg, bottom_bg, left_bg, right_bg)
        if inner_std < bg_min * 1.5 or inner_std < 8.0:
            return []

        return [
            SplitLine(position=float(x1) / w, orientation="vertical"),
            SplitLine(position=float(x2) / w, orientation="vertical"),
            SplitLine(position=float(y1) / h, orientation="horizontal"),
            SplitLine(position=float(y2) / h, orientation="horizontal"),
        ]

    def _find_inset_by_contour(self, gray: np.ndarray, w: int, h: int) -> List[SplitLine]:
        """轮廓检测法（备选）：用 Canny + findContours 找矩形窗口。"""
        edges = cv2.Canny(gray, 30, 100)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_area = w * h
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < img_area * 0.05 or area > img_area * 0.7:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) != 4:
                continue
            x, y, cw, ch = cv2.boundingRect(approx)
            if cw < w * 0.15 or ch < h * 0.15:
                continue
            if self._inset_differs(gray, x, y, cw, ch, w, h):
                return [
                    SplitLine(position=float(x) / w, orientation="vertical"),
                    SplitLine(position=float(x + cw) / w, orientation="vertical"),
                    SplitLine(position=float(y) / h, orientation="horizontal"),
                    SplitLine(position=float(y + ch) / h, orientation="horizontal"),
                ]
        return []

    def _find_inset_by_frame_diff(self, video_path: str, fps: float, w: int, h: int) -> List[SplitLine]:
        """帧差法检测屏中屏（对动态窗口 + 静态背景最有效）。

        原理：屏中屏内的有害视频是动态的，会产生帧间差异；
        背景是静态的，没有帧间差异。通过块差异分析定位窗口。

        适用场景：方差分析失效（窗口边缘软化或背景有纹理），
        但窗口内容是动态视频时，帧差法能精确定位。
        """
        if fps <= 0 or w == 0 or h == 0:
            return []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 4:
            cap.release()
            return []

        # 均匀采样 6 帧
        n = 6
        indices = np.linspace(0, total - 1, n, dtype=int)
        gray_frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                gray_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        cap.release()

        if len(gray_frames) < 3:
            return []

        # 累加相邻帧差异
        diff_sum = np.zeros((h, w), dtype=np.float32)
        for i in range(len(gray_frames) - 1):
            diff = cv2.absdiff(gray_frames[i], gray_frames[i + 1])
            diff_sum += diff.astype(np.float32)
        diff_avg = diff_sum / max(len(gray_frames) - 1, 1)

        # 块差异分析（12x12 网格）
        gh, gw = 12, 12
        bh, bw = h // gh, w // gw
        if bh < 10 or bw < 10:
            return []

        diff_map = np.zeros((gh, gw), dtype=np.float32)
        for i in range(gh):
            for j in range(gw):
                block = diff_avg[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
                diff_map[i, j] = float(block.mean())

        # 阈值：整体差异的 2 倍（窗口差异应显著高于背景）
        global_diff = float(diff_avg.mean())
        threshold = max(global_diff * 2.0, 5.0)

        binary = (diff_map > threshold).astype(np.uint8)

        # 形态学开运算
        kernel = np.ones((2, 2), dtype=np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.dilate(binary, kernel, iterations=1)

        # 找最大连通组件
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if num_labels <= 1:
            return []

        max_area = 0
        max_label = 0
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area > max_area:
                max_area = area
                max_label = label

        if max_label == 0 or max_area < 4:
            return []

        binary = (labels == max_label).astype(np.uint8)
        rows = np.any(binary, axis=1)
        cols = np.any(binary, axis=0)
        r_min, r_max = np.where(rows)[0][[0, -1]]
        c_min, c_max = np.where(cols)[0][[0, -1]]

        x1 = int(c_min * bw)
        y1 = int(r_min * bh)
        x2 = int((c_max + 1) * bw)
        y2 = int((r_max + 1) * bh)
        cw = x2 - x1
        ch = y2 - y1

        area_ratio = (cw * ch) / (w * h)
        if area_ratio < 0.03 or area_ratio > 0.7:
            return []

        return [
            SplitLine(position=float(x1) / w, orientation="vertical"),
            SplitLine(position=float(x2) / w, orientation="vertical"),
            SplitLine(position=float(y1) / h, orientation="horizontal"),
            SplitLine(position=float(y2) / h, orientation="horizontal"),
        ]

    @staticmethod
    def _inset_differs(gray: np.ndarray, x: int, y: int, cw: int, ch: int,
                       w: int, h: int, diff_threshold: float = 0.3) -> bool:
        """判断矩形窗口内部与外部的内容是否有显著差异。"""
        inner = gray[y:y + ch, x:x + cw]
        # 外部：取四周边缘带
        outer_top = gray[:max(1, y - 5), :]
        outer_bottom = gray[min(h - 1, y + ch + 5):, :]
        outer_left = gray[:, :max(1, x - 5)]
        outer_right = gray[:, min(w - 1, x + cw + 5):]
        outer_parts = [p for p in [outer_top, outer_bottom, outer_left, outer_right] if p.size > 0]
        if not outer_parts:
            return False
        outer = np.concatenate([p.flatten() for p in outer_parts])

        hist_in = cv2.calcHist([inner], [0], None, [32], [0, 256]).flatten()
        hist_out = cv2.calcHist([outer], [0], None, [32], [0, 256]).flatten()
        hist_in = hist_in / max(hist_in.sum(), 1e-12)
        hist_out = hist_out / max(hist_out.sum(), 1e-12)
        diff = 0.5 * float(np.sum((hist_in - hist_out) ** 2 / (hist_in + hist_out + 1e-12)))
        return diff > diff_threshold

    @staticmethod
    def _regions_differ(gray: np.ndarray, line: SplitLine, w: int, h: int,
                        diff_threshold: float = 0.3) -> bool:
        """判断分割线两侧区域的灰度直方图是否有显著差异。

        使用卡方距离衡量两侧直方图差异，阈值 0.3 表示显著不同。
        避免渐变背景中的伪直线被误判为分屏。
        """
        margin = 5  # 跳过直线本身
        if line.orientation == "vertical":
            x = int(line.position * w)
            left = gray[:, :max(1, x - margin)]
            right = gray[:, min(w - 1, x + margin):]
            if left.size == 0 or right.size == 0:
                return False
            hist_l = cv2.calcHist([left], [0], None, [32], [0, 256]).flatten()
            hist_r = cv2.calcHist([right], [0], None, [32], [0, 256]).flatten()
        else:
            y = int(line.position * h)
            top = gray[:max(1, y - margin), :]
            bottom = gray[min(h - 1, y + margin):, :]
            if top.size == 0 or bottom.size == 0:
                return False
            hist_l = cv2.calcHist([top], [0], None, [32], [0, 256]).flatten()
            hist_r = cv2.calcHist([bottom], [0], None, [32], [0, 256]).flatten()

        # 归一化为概率分布
        hist_l = hist_l / max(hist_l.sum(), 1e-12)
        hist_r = hist_r / max(hist_r.sum(), 1e-12)
        # 卡方距离（归一化到 [0, 2]）
        diff = 0.5 * float(np.sum((hist_l - hist_r) ** 2 / (hist_l + hist_r + 1e-12)))
        return diff > diff_threshold

    @staticmethod
    def _dedupe_lines(lines: List[SplitLine], tol: float = 0.05) -> List[SplitLine]:
        """去重相近的直线。"""
        out: List[SplitLine] = []
        for ln in lines:
            if all(abs(ln.position - o.position) > tol or ln.orientation != o.orientation for o in out):
                out.append(ln)
        return out

    # ─────────────── A.5 变速检测 ───────────────

    def _detect_speed(self, video_path: str, frames: List[np.ndarray], fps: float, has_audio: bool
                      ) -> Tuple[bool, float, str]:
        """视觉光流 + 音频（可选）双通道变速检测。MVP 仅视觉。"""
        if len(frames) < 2 or fps <= 0:
            return False, 1.0, "none"

        # 视觉通道：相邻帧 Farneback 光流均值
        flows = []
        prev = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        for nxt in frames[1:]:
            nxt_gray = cv2.cvtColor(nxt, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev, nxt_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag = np.hypot(flow[..., 0], flow[..., 1])
            flows.append(float(mag.mean()))
            prev = nxt_gray

        if not flows:
            return False, 1.0, "none"
        mean_flow = float(np.mean(flows))
        std_flow = float(np.std(flows))

        # 变速视频的真正特征：光流的不连续性（跳帧导致运动突变）
        # - 静态视频：光流均值低 + 标准差低（一致地小）→ 正常，不是变速
        # - 正常运动视频：光流均值中等 + 标准差中等 → 正常
        # - 加速视频（跳帧）：光流忽大忽小 → 标准差/均值比高（不连续）
        # - 极端加速：光流均值极大（每帧跨越大量原始时间）

        # 静态视频过滤：光流稳定地小，判定为正常静态视频
        # 这是修复真实视频误报的关键（0001/0050/0100/0150 都是静态画面）
        is_static = mean_flow < 1.0 and std_flow < max(0.5, mean_flow * 0.8)
        if is_static:
            return False, 1.0, "visual"

        # 变速检测：光流极大 或 不连续性高
        # 不连续性 = 标准差 / 均值，跳帧会让这个比值显著升高
        inconsistency = std_flow / max(mean_flow, 0.5)
        abnormal = mean_flow > 20.0 or (mean_flow > 1.0 and inconsistency > 2.5)
        if not abnormal:
            return False, 1.0, "visual"

        # 估算速度比
        if mean_flow > 20.0:
            # 光流极大 → 明显加速
            ratio = min(3.0, mean_flow / 8.0)
        else:
            # 不连续性高 → 跳帧加速
            ratio = min(3.0, max(1.5, inconsistency))

        # 音频通道（暂未实现，留接口）
        channel = "visual"
        if has_audio:
            # TODO: 音频频谱分析语速
            channel = "both"
        return True, float(ratio), channel
