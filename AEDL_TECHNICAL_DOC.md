# AEDL 技术文档

## 抗规避检测层（Adversarial Evasion Detection Layer）

---

## 1. 概述

AEDL 是为 NSFW Detector 视频审核系统设计的**抗规避检测中间件**。它在前端与原后端审核 API 之间插入一层处理逻辑，专门识别和还原上传者可能使用的规避手段，确保违规内容无法通过变换绕过审核。

### 1.1 设计目标

- **零前端改动**：保持原 `/api/v1/detect` 的 `{detection, report}` 顶层结构契约不变
- **保守不漏检**：任何信号源揭示的违规都不应被遗漏
- **不污染类别**：避免变换后模型误判的类别分数污染最终结果
- **可配置可校准**：所有阈值可通过 `configs/aedl.yaml` 调整

### 1.2 部署架构

```
前端 ──> AEDL (port 8001) ──> 原 NSFW Detector (port 8000)
         /api/v1/aedl/detect     /api/v1/detect
```

前端只需将请求端点从 `/api/v1/detect` 改为 `/api/v1/aedl/detect`，其余请求/响应格式完全兼容。

---

## 2. 五阶段流水线架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    AEDL Pipeline                                 │
│                                                                  │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌────┐ │
│  │ A. 感知 │──>│ B. 路由 │──>│ C. 变换 │──>│ 后端API │──>│ D. │──> E. 融合
│  │  层     │   │  层     │   │  还原层 │   │ /detect │   │一致性│
│  └─────────┘   └─────────┘   └─────────┘   └─────────┘   └────┘ │
│                                                                  │
│  输入视频 ────────────────────────────────────────────> 融合输出   │
└─────────────────────────────────────────────────────────────────┘
```

### 2.1 模块 A：输入感知层

**位置**：`aedl/input_perception.py`

**职责**：对视频首帧进行多维度异常检测，毫秒级 CPU 算子。

**检测能力（8 种）**：

| 检测项 | 方法 | 字段 | 阈值 |
|--------|------|------|------|
| 水平镜像 | SSIM 对称性 | `is_mirror_suspicious` | `mirror_ssim_threshold: 0.95` |
| 垂直翻转 | SSIM 垂直对称性 | `is_vertical_flip_suspicious` | `vflip_ssim_threshold: 0.90` |
| 边框遮挡 | 四边方差检测 | `is_border_occluded` | `border_variance_threshold: 50.0` |
| 画质降质 | 拉普拉斯方差+块效应 | `is_quality_degraded` | `blocking_artifact_threshold: 0.5` |
| 分屏 | Canny+Hough+直方图差异 | `is_split_screen` | `split_line_min_ratio: 0.8` |
| 变速 | 光流不连续性 | `is_speed_abnormal` | `inconsistency > 2.5` |
| 亮度异常 | 平均亮度 | `is_darkened` | `dark_brightness_threshold: 40.0` |
| 闪烁插入 | 帧间亮度突变 | `is_flickering` | `flicker_brightness_threshold: 25.0` |

**采样策略**：仅检测前 3 个非黑帧（`max_frames_to_check: 3`）。

### 2.2 模块 B：策略路由层

**位置**：`aedl/strategy_router.py`

**职责**：根据感知层异常信号决策要执行的变换组合。

**两层变换链生成机制**：

#### 保险还原变换（无条件追加）

无视感知层是否检测到异常，默认对所有视频追加以下变换链送审：

```yaml
router:
  force_restore_transforms:
    - mirror      # 水平翻转还原
    - vflip       # 垂直翻转还原
    - brighten    # 亮度增强还原
```

**原理**：感知层可能漏检（如不对称画面的镜像），保险机制确保所有常见规避手段都有对应的还原版本送审。由模块 D 通过分数对比判断是否真的存在规避。

#### 异常驱动变换链

根据感知层检测到的异常，生成对应的还原变换：

| 感知异常 | 变换 | 参数 |
|---------|------|------|
| `is_mirror_suspicious` | `mirror` | - |
| `is_vertical_flip_suspicious` | `vflip` | - |
| `is_border_occluded` | `center_crop` | `ratio = 1 - border_width_ratio` |
| `is_quality_degraded` | `brighten` + `denoise` | 串联 |
| `is_darkened` | `brighten` | - |
| `is_split_screen` | `split_extract` | 基于分割线 |
| `is_speed_abnormal` | `resample` | `ratio = estimated_speed_ratio` |
| `is_flickering` | `brighten` | - |

**版本数量限制**：`max_restored_versions: 4`（含原始共 5 版本）

### 2.3 模块 C：变换还原层

**位置**：`aedl/transforms.py`

**职责**：对原始视频施加可逆或语义保持变换，生成候选还原版本。

**支持的变换（7 种）**：

| 变换 | 方法 | 用途 |
|------|------|------|
| `mirror` | `cv2.flip(frame, 1)` | 水平翻转还原 |
| `vflip` | `cv2.flip(frame, 0)` | 垂直翻转还原 |
| `center_crop` | 中心裁剪 | 边框遮挡还原 |
| `brighten` | Gamma校正+CLAHE | 画质降质/亮度异常还原 |
| `denoise` | 高斯/非局部均值去噪 | 压缩伪影还原 |
| `split_extract` | 子画面提取 | 分屏还原 |
| `resample` | 帧重复/跳帧 | 变速还原 |

**变速还原策略**：
- 加速还原（`ratio > 1`）：每帧重复 `ratio` 次，恢复原始时长
- 减速还原（`ratio < 1`）：按 `skip_period` 跳过部分帧，恢复原始节奏
- 保持原始 fps，不改变抽帧密度

### 2.4 模块 D：一致性校验层

**位置**：`aedl/consistency.py`

**职责**：对比原始版本与各还原版本的七维概率向量，计算规避嫌疑分。

**七维类别标签**（与原 API 一致）：
```
["Smoke", "Blood", "Violent", "Abusive", "Sexy", "Money", "Policy"]
```

**长短标签映射**：
| 短标签 | 长标签（is_harmful=true 时返回） |
|--------|-------------------------------|
| Smoke | A person smoking a cigarette |
| Blood | Blood on the ground, bloody scene, gore |
| Violent | People fighting and hitting each other |
| Abusive | Person making aggressive gestures |
| Sexy | Sexually suggestive content and exposure |
| Money | Displaying large amounts of cash suspiciously |
| Policy | Politically sensitive content and symbols |

**一致性度量**：

1. **单标签跃升**：`delta = restored_score - original_score`
   - 触发阈值：`label_jump_threshold: 0.4`
   - 断崖阈值：`label_cliff_threshold: 0.5` + `restored_high_confidence: 0.8`

2. **Jensen-Shannon 散度**：衡量分布偏移
   - 阈值：`jsd_threshold: 0.15`

3. **规避置信度评分**：
   ```python
   score = 0.35 * max_jump + 0.25 * (mean_jsd / ln2) + 0.25 * trigger_flag + 0.15 * anomaly_norm
   ```

4. **风险分级**：
   | 等级 | 阈值 |
   |------|------|
   | high | `>= 0.7` |
   | medium | `>= 0.5` |
   | low | `>= 0.2` |
   | none | `< 0.2` |

### 2.5 模块 E：结果融合输出

**位置**：`aedl/fusion.py`

**职责**：融合原始+还原版本的检测结果，保持原 API 契约结构。

**关键设计原则**：

#### 分数融合策略（v4 设计）

1. **category_scores 只来自原 API 的 category_scores 字段**
   - `harmful_segments[].score` **不投影到标签**（segment 的 category 可能是模型误判）
   - `calibrated_score` **不投影到标签**（它是总体违规分，不对应特定类别）

2. **触发跃升才融合还原版本分数**
   ```python
   triggered_labels = {d.label for d in consistency.trigger_details if d.flagged}
   final_scores = _fuse_scores(reports, triggered_labels)
   ```
   - 触发跃升的标签：取 `max(原始, 还原版本)` 高分
   - 未触发的标签：保留原始版本真实分数
   - **目的**：避免 center_crop 裁剪后构图变化导致的模型误判污染最终结果

3. **anomaly_score 融合**：
   ```python
   fused_anomaly = max(
       max_calibrated_score,   # 最准确的总体违规分
       max_anomaly_score,      # 原 API 的 anomaly_score
       max_category_score      # category_scores 最大值
   )
   ```
   - 原始版本始终纳入保底
   - 还原版本只在有触发跃升时纳入

4. **predicted_categories**：只基于 `category_scores >= 0.5`
   - 不继承原 API 的 predicted_categories 误判
   - 不投影 harmful_segments 到标签

#### 顶层响应结构（方案 A）

```json
{
  "detection": { ... },           // 与原 API 完全兼容
  "report": { ... },              // 与原 API 完全兼容
  "aedl_analysis": {              // 可选附加字段
    "evasion_score": 0.0,
    "evasion_level": "none|low|medium|high",
    "evasion_type": "mirror_flip|vertical_flip|border_occlusion|...",
    "anomalies_detected": [...],
    "trigger_details": [...],
    "evidence_chain": "...",
    "needs_manual_review": false,
    "review_priority": "none|low|medium|high",
    "processing_metadata": {
      "versions_generated": 3,
      "model_inference_count": 4,
      "total_processing_time_ms": 10283.66,
      "perception_time_ms": 1175.77,
      "transform_time_ms": 4090.48,
      "inference_time_ms": 5012.04,
      "consistency_time_ms": 0
    },
    "restored_reports": [ ... ]   // 各还原版本的完整报告
  }
}
```

---

## 3. 检测能力详解

### 3.1 镜像规避检测

**问题**：上传者水平翻转视频，让模型训练数据中的特征失效。

**检测**：
- 感知层：SSIM 对称性（阈值 0.95）
- 保险机制：无条件追加 `mirror` 变换链送审

**还原**：`cv2.flip(frame, 1)` 水平翻转还原

### 3.2 垂直翻转检测

**问题**：上传者上下翻转视频，绕过水平镜像检测。

**检测**：
- 感知层：SSIM 垂直对称性（阈值 0.90）
- 保险机制：无条件追加 `vflip` 变换链送审

**还原**：`cv2.flip(frame, 0)` 垂直翻转还原

### 3.3 边框遮挡检测

**问题**：上下加黑边/贴纸挡住关键画面。

**检测**：分别检测四条边（上/下/左/右），任一条边方差低 + 内部方差显著高 → 遮挡。

**还原**：`center_crop` 中心裁剪，裁剪比例 `1 - border_width_ratio`，约束在 `[0.80, 0.95]`。

### 3.4 画质降质检测

**问题**：故意压低画质让模型看不清。

**检测**：拉普拉斯方差（模糊度）+ 8×8 块效应（压缩伪影）。

**还原**：`brighten`（Gamma校正+CLAHE）+ `denoise`（高斯/非局部均值去噪）串联。

### 3.5 分屏检测

**问题**：把违规内容塞进小窗混在正常视频里。

**检测**：Canny边缘 + Hough直线检测 + 卡方直方图距离验证两侧内容差异（阈值 0.3）。

**还原**：`split_extract` 提取最大子画面。

### 3.6 变速检测

**问题**：加速/减速让模型抽帧时错过违规片段。

**检测**：Farneback光流，静态视频过滤（`mean_flow < 1.0 and std_flow < max(0.5, mean_flow * 0.8)`），不连续性检测（`std_flow / mean_flow > 2.5`）。

**还原**：`resample` 帧重复（加速还原）或跳帧（减速还原），保持原始 fps。

### 3.7 亮度异常检测

**问题**：蒙版遮挡/调暗/半透明化导致模型看不清违规内容。

**检测**：平均亮度低于 `dark_brightness_threshold: 40.0`。

**还原**：`brighten` Gamma校正+CLAHE 增强亮度。

### 3.8 闪烁插入检测

**问题**：在无害视频中插入几帧有害内容，模型抽帧可能错过。

**检测**：均匀采样 12 帧计算亮度，帧间亮度突变超过阈值且突变/均值比 > 3.0。

**还原**：`brighten` 增强亮度尝试还原有害帧内容。

---

## 4. API 接口

### 4.1 AEDL 检测端点

```
POST /api/v1/aedl/detect
Content-Type: multipart/form-data

字段:
  - file: 视频文件 (mp4, avi, mov)
  - uploader (可选): 上传者标识，用于白名单

响应:
{
  "detection": { ... },          // 与原 API 兼容
  "report": { ... },             // 与原 API 兼容
  "aedl_analysis": { ... }       // AEDL 附加分析
}
```

### 4.2 预览端点

```
POST /api/v1/aedl/preview
```

仅返回感知层和路由层结果，不调用后端模型。用于快速调试感知/路由逻辑。

### 4.3 健康检查

```
GET /health
```

---

## 5. 配置参数

**配置文件**：`configs/aedl.yaml`

所有阈值均可通过修改配置文件调整，无需改代码。详见配置文件注释。

### 5.1 关键配置项

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `perception.mirror_ssim_threshold` | 0.95 | 水平镜像对称性阈值 |
| `perception.vflip_ssim_threshold` | 0.90 | 垂直翻转对称性阈值 |
| `perception.dark_brightness_threshold` | 40.0 | 亮度异常阈值 |
| `perception.flicker_brightness_threshold` | 25.0 | 闪烁亮度突变阈值 |
| `router.force_restore_transforms` | `[mirror, vflip, brighten]` | 保险还原变换列表 |
| `router.max_restored_versions` | 4 | 最大还原版本数 |
| `consistency.label_jump_threshold` | 0.4 | 标签跃升触发阈值 |
| `consistency.evasion_weights` | `[0.35, 0.25, 0.25, 0.15]` | 规避分权重 |

---

## 6. 限制与已知问题

1. **镜像检测对天然对称内容失效**：SSIM 方法无法区分人工镜像与天然对称。已通过保险机制（无条件追加 mirror 变换链）缓解。

2. **半透明化规避难以检测**：半透明混合有害与无害内容，亮度可能不降低，感知层难以识别。依赖保险机制（brighten 还原）。

3. **原模型能力是上限**：AEDL 只能还原变换，不能提升原 NSFW Detector 模型本身的能力。如果还原后模型仍判定为安全，AEDL 无法改变结果。

4. **音频通道未实现**：变速检测仅视觉通道，音频频谱分析为 TODO。

5. **处理时间增加**：每个视频生成 3-4 个还原版本，模型推理次数增加，总处理时间约为原来的 4-5 倍。

---

## 7. 文件结构

```
ocvpre/
├── aedl/
│   ├── __init__.py
│   ├── config.py              # 配置加载
│   ├── schemas.py             # Pydantic 数据模型
│   ├── input_perception.py    # 模块 A：输入感知层
│   ├── strategy_router.py     # 模块 B：策略路由层
│   ├── transforms.py          # 模块 C：变换还原层
│   ├── backend_client.py      # 原 API 客户端
│   ├── consistency.py         # 模块 D：一致性校验层
│   ├── fusion.py              # 模块 E：结果融合输出
│   ├── pipeline.py            # 流水线编排器
│   └── server.py              # FastAPI 服务入口
├── configs/
│   └── aedl.yaml              # 配置文件
└── AEDL_TECHNICAL_DOC.md      # 本文档
```
