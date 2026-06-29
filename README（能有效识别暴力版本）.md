# AEDL 对抗规避检测层

> Adversarial Evasion Detection Layer — 部署于 NSFW Detector 视频审核系统上游的独立中间模块。

检测内容创作者常用的低成本规避手段（镜像翻转、垂直翻转、边缘遮挡、画质降质、画面分屏/屏中屏、播放变速、暗光蒙版、闪烁插入帧），通过「检测—过滤—变换还原—交叉验证—一致性度量」五段式机制识别并还原被刻意处理过的输入，将模型有效覆盖范围从「理想样本」扩展到「对抗样本」。

> **保险机制**：默认对所有视频追加 `mirror` / `vflip` / `brighten` 三类还原变换，由模块 D 通过分数对比判断是否真的存在规避，避免感知层漏检导致规避内容绕过。

---

## 设计目标

- **零数据增强**：仅基于 OpenCV / NumPy 传统 CV 算子，不依赖标注数据，不引入模型训练
- **模型无关性**：作为独立预处理模块，与下游审核模型完全解耦，可复用
- **保守召回策略**：任一还原版本揭示的违规信号均不会被遗漏
- **可解释证据链**：输出具体规避类型与量化证据，支撑人工复核
- **吞吐量可控**：快速通道优先，仅 10%-15% 视频触发完整流程

---

## 系统架构

```
[前端上传视频]
       │
       ▼
┌────────────────────────────┐
│  AEDL 中间模块 (端口 8001)  │
│                            │
│  模块 A: 输入感知层          │  ← 快速 CV 算子，<50ms
│      ↓                    │
│  模块 B: 策略路由层          │  ← 决策表驱动 + 白名单
│      ↓                    │
│  模块 C: 变换还原层          │  ← 生成最多 8 个还原版本
│      ↓                    │
│  调用原 /api/v1/detect     │  ← 原始+还原版本并行送审
│      ↓                    │
│  模块 D: 一致性校验层        │  ← JSD / 单标签跃升 / 规避分
│      ↓                    │
│  模块 E: 结果融合输出        │  ← 保守最大值 + 证据链
└────────────────────────────┘
       │
       ▼
[融合响应返回前端]
```

**关键点**：AEDL 不替代原审核模型，而是作为前置中间层，复用原 `/api/v1/detect` 接口对多版本分别推理，再由模块 D/E 对多版本输出做一致性度量和融合。

---

## 目录结构

```
nsfw/
├── aedl/
│   ├── __init__.py            # 包入口
│   ├── config.py              # 配置加载（YAML + 环境变量）
│   ├── schemas.py             # Pydantic 数据模型
│   ├── input_perception.py    # 模块 A：输入感知层
│   ├── strategy_router.py     # 模块 B：策略路由层
│   ├── transforms.py          # 模块 C：变换还原层
│   ├── backend_client.py      # 后端客户端（调用原 API）
│   ├── consistency.py         # 模块 D：一致性校验层
│   ├── fusion.py              # 模块 E：结果融合输出
│   ├── pipeline.py            # 主流程编排
│   ├── server.py              # FastAPI 服务入口
│   └── requirements.txt       # 依赖清单
├── configs/
│   ├── aedl.yaml              # AEDL 主配置
│   └── whitelist.txt          # 上传者白名单
├── API.md                     # 原 NSFW Detector API 文档
└── 方案一_对抗规避检测层_完善版.txt  # 设计方案原文
```

---

## 安装

### 环境要求

- Python 3.10+
- 后端已运行 NSFW Detector 服务（默认 `http://localhost:8000`）

### 步骤

```bash
# 1. 安装依赖
pip install -r aedl/requirements.txt

# 2.（可选）编辑配置
# 修改 configs/aedl.yaml 中的 backend.base_url 指向你的 NSFW Detector 服务

# 3. 启动 AEDL 服务
python -m aedl.server
# 或开发模式
uvicorn aedl.server:app --host 0.0.0.0 --port 8001 --reload
```

启动后访问：
- Swagger UI：`http://localhost:8001/docs`
- 健康检查：`http://localhost:8001/api/v1/aedl/health`

---

## API 接口

### `POST /api/v1/aedl/detect`

上传视频执行完整 AEDL 流程。

#### 前端对接说明（重要）

**响应结构完全兼容原 `/api/v1/detect`**：

- 顶层字段保持原 NSFW Detector 的 `detection` + `report` 结构不变
- 前端原有渲染代码（`data.detection.xxx` / `data.report.xxx`）**无需任何修改**
- 前端只需把请求端点从 `/api/v1/detect` 改成 `/api/v1/aedl/detect`（请求参数完全一致）
- 新增可选字段 `aedl_analysis`，前端可渐进式接入规避分析展示，不渲染也不影响原有功能

#### 请求

`multipart/form-data`

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `file` | file | 是 | 视频文件（mp4/avi/mov/mkv/wmv/flv/webm） |
| `threshold` | float | 否 | 临时异常阈值 [0,1]，覆盖配置默认值 |
| `uploader` | string | 否 | 上传者标识（用于白名单匹配） |
| `keep_temp` | bool | 否 | 保留还原视频用于调试（默认 false） |

> `file` 与 `threshold` 与原 `/api/v1/detect` 完全一致，`uploader` / `keep_temp` 为 AEDL 新增可选参数。

#### 响应：`AEDLResponse`

顶层 `{detection, report}` 与原 API 完全一致，新增 `aedl_analysis` 可选字段：

```json
{
  "detection": {
    "video_id": "test",
    "duration": 15.42,
    "is_harmful": true,
    "anomaly_score": 0.85,
    "predicted_categories": ["Smoke"],
    "category_scores": {"Smoke": 0.85, "Blood": 0.12, "...": 0.0},
    "harmful_segments": [
      {"start_time": 0.0, "end_time": 2.5, "score": 0.91, "category": "吸烟", "category_en": "Smoke"}
    ],
    "keyframe_urls": ["/api/v1/keyframes/keyframe_test_0.0_2.5.jpg?report_id=..."]
  },
  "report": {
    "report_id": "...",
    "video_id": "test",
    "scan_time": "2026-06-28 10:00:00",
    "alert_level": "HIGH",
    "anomaly_score": 0.85,
    "harmful_contents": [
      {
        "category_en": "Smoke",
        "category_zh": "吸烟",
        "confidence": 0.85,
        "time_segments": "0.0-2.5s",
        "keyframe_path": "...",
        "description": "检测到疑似吸烟行为，可能违反平台内容规范"
      }
    ],
    "summary": "检测到异常评分 0.85，涉及类别：Smoke；规避分析：检测到变换【mirror】后 Smoke 标签置信度从 0.18 跃升至 0.83",
    "action_suggestion": "建议立即下架并转人工审核，等待进一步处理；检测到高风险规避嫌疑，强制人工复核",
    "processing_time": 0.0032
  },
  "aedl_analysis": {
    "evasion_score": 0.72,
    "evasion_level": "high",
    "evasion_type": "mirror_flip",
    "anomalies_detected": ["mirror_symmetry"],
    "trigger_details": [
      {
        "transform": "mirror",
        "label": "Smoke",
        "original_score": 0.18,
        "restored_score": 0.83,
        "delta": 0.65,
        "flagged": true
      }
    ],
    "evidence_chain": "检测到变换【mirror】后 Smoke 标签置信度从 0.18 跃升至 0.83，跃升幅度 0.65；疑似镜像规避上传",
    "needs_manual_review": true,
    "review_priority": "high",
    "processing_metadata": {
      "versions_generated": 1,
      "model_inference_count": 2,
      "total_processing_time_ms": 1250.5,
      "perception_time_ms": 42.3,
      "transform_time_ms": 380.1,
      "inference_time_ms": 820.4,
      "consistency_time_ms": 0.5
    },
    "restored_reports": [
      {"version_id": "restored_1", "transforms": ["mirror"], "report": { ... }}
    ]
  }
}
```

> `detection` / `report` 字段结构与原 `/api/v1/detect` 完全一致，字段值已融合原始 + 还原版本的保守最大值。
> `aedl_analysis` 为新增可选字段，前端不渲染也不影响原有检测结果的展示。

### `GET /api/v1/aedl/health`

健康检查，返回 AEDL 自身状态与后端 NSFW Detector 可用性。

---

## 调用示例

**curl**：
```bash
curl -X POST http://localhost:8001/api/v1/aedl/detect \
  -F "file=@test.mp4" \
  -F "threshold=0.5" \
  -F "uploader=user_001"
```

**JavaScript（fetch）**：
```javascript
const formData = new FormData();
formData.append("file", fileInput.files[0]);
formData.append("threshold", "0.5");
const resp = await fetch("http://localhost:8001/api/v1/aedl/detect", {
  method: "POST",
  body: formData,
});
const data = await resp.json();

// 原有渲染逻辑完全不变
console.log(data.detection.anomaly_score);        // 融合后的异常分数
console.log(data.report.alert_level);             // 融合后的预警等级
console.log(data.detection.keyframe_urls[0]);     // 关键帧 URL

// 可选：渲染规避分析（新增字段，不影响原有逻辑）
if (data.aedl_analysis) {
  console.log(data.aedl_analysis.evasion_level);  // "high" / "medium" / "low" / "none"
  console.log(data.aedl_analysis.evidence_chain); // 证据链文本
}
```

**Python**：
```python
import httpx

with open("test.mp4", "rb") as f:
    resp = httpx.post(
        "http://localhost:8001/api/v1/aedl/detect",
        files={"file": ("test.mp4", f, "video/mp4")},
        data={"threshold": "0.5"},
        timeout=300,
    )
data = resp.json()
# 原有字段访问完全不变
print(data["detection"]["anomaly_score"], data["report"]["alert_level"])
# 可选：规避分析
if data.get("aedl_analysis"):
    print(data["aedl_analysis"]["evasion_level"], data["aedl_analysis"]["evasion_score"])
```

---

## 模块详解

### 模块 A：输入感知层（[input_perception.py](aedl/input_perception.py)）

仅对视频前 3 个非黑帧执行毫秒级 CV 检测：

| 检测项 | 方法 | 输出 |
|--------|------|------|
| 镜像对称性 | SSIM 比对水平翻转帧 | `is_mirror_suspicious` + `symmetry_score` |
| 垂直翻转 | SSIM 比对垂直翻转帧（阈值 0.90） | `is_vertical_flip_suspicious` + `vertical_symmetry_score` |
| 边缘遮挡 | 四边像素带方差 + 内缘梯度 | `is_border_occluded` + `border_width_ratio` |
| 画质降质 | 拉普拉斯方差 + 压缩块效应 | `is_quality_degraded` + `laplacian_score` |
| 分屏结构 | Canny + Hough 直线检测 | `is_split_screen` + `split_lines` |
| 屏中屏检测 | 块方差分析（12×12 网格 + 形态学开运算 + 连通域）+ 轮廓检测 | `is_split_screen`（inset 模式）+ `split_lines`（2 垂直 + 2 水平边） |
| 变速异常 | Farneback 光流（视觉通道） | `is_speed_abnormal` + `estimated_speed_ratio` |
| 暗光蒙版 | 平均亮度低于 40 判定异常 | `is_darkened` + `mean_brightness` |
| 闪烁插入帧 | 12 帧采样，帧间亮度突变 > 25 | `is_flickering` + `flicker_score` |

> **屏中屏检测原理**：将灰度图划分为 12×12 块，计算每块方差；高方差区域（视频内容）被低方差区域（均匀背景）包围即判定为屏中屏。形态学开运算去除孤立噪声，连通组件分析定位窗口边界。配合 Canny+findContours 轮廓检测作为备选方案，覆盖软边缘和清晰边框两种场景。

### 模块 B：策略路由层（[strategy_router.py](aedl/strategy_router.py)）

决策表驱动的异常→变换映射：

| 检测到的异常 | 触发的变换 |
|--------------|------------|
| 镜像对称性异常 | 水平镜像还原（M） |
| 垂直翻转异常 | 垂直镜像还原（V） |
| 边缘遮挡 | 中心裁剪还原（C） |
| 画质降质 | 亮度增强（B）+ 去噪（D）串联 |
| 分屏结构 | 分屏主画面提取（S） |
| 分屏/屏中屏 | 3 种组合变换链（见下文） |
| 暗光蒙版 | 亮度增强（B） |
| 闪烁插入帧 | 亮度增强（B） |
| 帧率/变速异常 | 抽帧密度补偿（F） |
| 多项叠加 | 按置信度选前 2 项串联 |
| 保险机制 | 默认追加 `mirror` / `vflip` / `brighten`，所有视频均送审 |

**分屏/屏中屏组合变换链**（每条都先提取窗口并放大到原始尺寸，再叠加不同图像增强）：

1. `split_extract+brighten`：提取放大 + 亮度增强
2. `split_extract+sharpen+brighten`：提取放大 + Unsharp Mask 锐化 + 亮度增强
3. `split_extract+contrast+brighten`：提取放大 + CLAHE 对比度增强 + 亮度增强

约束：最多生成 8 个还原版本（含原始共 9 版本），单链最多串联 2 项变换。命中白名单或低置信度时直接旁路。

### 模块 C：变换还原层（[transforms.py](aedl/transforms.py)）

| 变换 | 操作 | 适用场景 |
|------|------|----------|
| 水平镜像 | `cv2.flip(frame, 1)` | 镜像规避还原 |
| 垂直镜像 | `cv2.flip(frame, 0)` | 垂直翻转规避还原 |
| 中心裁剪 | 动态裁剪比例 [0.80, 0.95] | 去除遮挡边框 |
| 亮度增强 | Gamma(γ=0.8) + CLAHE | 还原暗光/蒙版细节 |
| Unsharp Mask 锐化 | `addWeighted(原图×1.5, 模糊×-0.5)` | 增强放大画面的边缘细节 |
| CLAHE 对比度增强 | LAB 空间 `clipLimit=4.0` | 让缩小画面的细节更突出 |
| 轻量去噪 | 高斯 / fastNlMeans | 还原降质画面 |
| 抽帧补偿 | 跳帧采样（零额外计算） | 变速视频 |
| 分屏提取 | 按直线切割取最大子画面 | 多画面拼接 |
| 屏中屏提取 | 2 垂直 + 2 水平边 → 提取中间矩形窗口 | 缩小有害画面嵌入无害背景 |
| 提取后放大 | `cv2.resize` INTER_LANCZOS4 放大到原始尺寸 | 让模型看到的画面尺寸与正常视频一致 |
| 组合变换 | `split_extract+{brighten/sharpen+brighten/contrast+brighten}` | 分屏/屏中屏多种增强组合，最大化检测率 |

### 模块 D：一致性校验层（[consistency.py](aedl/consistency.py)）

对原始与还原版本的七维类别分数向量进行：

1. **单标签跃升**：`Δ = P_restored(label) − P_original(label)`，Δ > 0.15 标记可疑
2. **断崖式跃升**：Δ > 0.3 且 `P_restored > 0.5`，高度疑似
3. **JSD 分布偏移**：Jensen-Shannon 散度 > 0.15 视为显著偏移
4. **多视角投票**：任一还原版本 `P > 0.5` 触发标记
5. **规避分加权**：
   ```
   EvasionScore = 0.35×max(Δ_label) + 0.25×mean(JSD)/ln(2)
                + 0.25×trigger_flag + 0.15×anomaly_count_normalized
   ```

风险分级：

| 范围 | 等级 | 处置 |
|------|------|------|
| < 0.2 | none | 正常放行 |
| 0.2 ~ 0.5 | low | 记录日志 |
| 0.5 ~ 0.7 | medium | 优先人工复核 |
| ≥ 0.7 | high | 强制人工复核 + 证据链 |

### 模块 E：结果融合输出（[fusion.py](aedl/fusion.py)）

采用**双融合策略**，平衡召回率与准确率：

- **`category_scores`（严格融合）**：
  - 类别分数的唯一来源是原 API 的 `category_scores` 字段
  - `harmful_segments[].score` 和 `calibrated_score` 均不投影到标签
  - 还原版本的分数只在「触发跃升」（`triggered_labels`）时才融合
  - 避免裁剪后构图变化导致的模型误判污染最终结果（如血腥视频被误标为辱骂）

- **`anomaly_score`（宽松融合，用 `calibrated_score` 兜底）**：
  - 原始版本的 `calibrated/anomaly_score` 始终纳入（保底）
  - 还原版本的分数在以下任一条件满足时纳入：
    - 有触发跃升的标签（`triggered_labels` 非空）
    - 还原版本 `is_harmful=true`（模型判定违规）
    - 还原版本 `calibrated_score > 原始 + 0.1`（显著提升）
  - 取所有纳入版本的 `max(calibrated, anomaly, max(category_scores))`

- **`predicted_categories`**：只基于 `category_scores >= 0.5`，不继承原 API 的 `predicted_categories` 误判

- **分屏/屏中屏强制机制**（原模型对缩小画面检测能力有限）：
  - 检测到 `split_screen` 异常时，强制将 `anomaly_score` 提升到 0.5（MEDIUM 等级），确保进入人工审核
  - 检测到 `split_screen` 异常时，强制 `needs_manual_review=true` 且 `priority=high`，不依赖原模型判断

- **证据链**：输出跃升幅度最大的标签及对应变换
- **透传原始报告**：保留原 `/api/v1/detect` 完整响应供前端使用

---

## 配置

主配置文件 [configs/aedl.yaml](configs/aedl.yaml) 集中管理所有阈值，对应方案文档「六、关键参数与阈值汇总」。

支持环境变量覆盖：

| 环境变量 | 说明 | 默认值 |
|----------|------|--------|
| `AEDL_CONFIG` | 配置文件路径 | `configs/aedl.yaml` |
| `AEDL_BACKEND_URL` | 后端 NSFW Detector 地址 | `http://localhost:8000` |
| `AEDL_PORT` | AEDL 服务端口 | `8001` |

白名单文件 [configs/whitelist.txt](configs/whitelist.txt) 每行一个上传者标识，`#` 开头为注释，命中即跳过全部 AEDL 检测。

---

## 依赖

| 库 | 用途 |
|----|------|
| fastapi + uvicorn | Web 服务 |
| httpx | 异步调用原 `/api/v1/detect` |
| opencv-python | 全部 CV 算子（SSIM/Canny/Hough/Laplacian/光流/CLAHE） |
| numpy | 数值计算与向量操作 |
| pydantic | 数据校验 |
| pyyaml | 配置加载 |

无 GPU 依赖，无需深度学习框架，可在 CPU 环境部署。

---

## 性能特征

以 1080p 60 秒视频为基准：

| 阶段 | 耗时 |
|------|------|
| 模块 A 输入感知 | < 50ms |
| 模块 B 策略路由 | < 1ms |
| 模块 C 变换还原 | 200-500ms / 版本（最多 8 版本） |
| 后端模型推理 | 取决于原模型（并行后取 max） |
| 模块 D 一致性校验 | < 1ms |
| 模块 E 结果融合 | < 1ms |

由于采用保险机制（默认追加 `mirror`/`vflip`/`brighten`）+ 分屏组合变换链，触发完整流程的视频会增加模型推理量。保险机制可按需在 [configs/aedl.yaml](configs/aedl.yaml) 中通过 `router.force_restore_transforms` 调整。

---

## 变更日志

| 日期 | 变更 |
|------|------|
| 2026-06-28 | 初版：实现方案一完整五段式架构，含 A/B/C/D/E 五模块与 FastAPI 服务入口 |
| 2026-06-28 | 新增垂直翻转、暗光蒙版、闪烁插入帧检测；启用保险机制（`mirror`/`vflip`/`brighten` 默认送审）；`max_restored_versions` 提升至 8 |
| 2026-06-28 | 新增屏中屏检测（块方差分析 + 形态学开运算 + 连通域 + 轮廓检测备选）；分屏提取后放大到原始尺寸（INTER_LANCZOS4）；新增 `_sharpen`（Unsharp Mask）与 `_contrast`（CLAHE `clipLimit=4.0`）变换；3 种分屏组合变换链 |
| 2026-06-28 | 双融合策略：`category_scores` 严格融合（仅 `triggered_labels`），`anomaly_score` 宽松融合（含 `is_harmful` / `calibrated` 显著提升条件）；分屏强制 `anomaly_score >= 0.5` 且强制人工审核；`predicted_categories` 改为基于 `category_scores >= 0.5`，不继承原 API 误判；阈值下调 `label_jump_threshold: 0.4→0.15`、`label_cliff_threshold: 0.5→0.3`、`restored_high_confidence: 0.8→0.5` |
