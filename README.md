# AEDL 对抗规避检测层

> Adversarial Evasion Detection Layer — 部署于 NSFW Detector 视频审核系统上游的独立中间模块。

检测内容创作者常用的低成本规避手段（镜像翻转、边缘遮挡、画质降质、画面分屏、播放变速），通过「检测—过滤—变换还原—交叉验证—一致性度量」五段式机制识别并还原被刻意处理过的输入，将模型有效覆盖范围从「理想样本」扩展到「对抗样本」。

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
│  模块 C: 变换还原层          │  ← 生成 1-3 个还原版本
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
| 边缘遮挡 | 四边像素带方差 + 内缘梯度 | `is_border_occluded` + `border_width_ratio` |
| 画质降质 | 拉普拉斯方差 + 压缩块效应 | `is_quality_degraded` + `laplacian_score` |
| 分屏结构 | Canny + Hough 直线检测 | `is_split_screen` + `split_lines` |
| 变速异常 | Farneback 光流（视觉通道） | `is_speed_abnormal` + `estimated_speed_ratio` |

### 模块 B：策略路由层（[strategy_router.py](aedl/strategy_router.py)）

决策表驱动的异常→变换映射：

| 检测到的异常 | 触发的变换 |
|--------------|------------|
| 镜像对称性异常 | 水平镜像还原（M） |
| 边缘遮挡 | 中心裁剪还原（C） |
| 画质降质 | 亮度增强（B）+ 去噪（D）串联 |
| 分屏结构 | 分屏主画面提取（S） |
| 帧率/变速异常 | 抽帧密度补偿（F） |
| 多项叠加 | 按置信度选前 2 项串联 |

约束：最多生成 3 个还原版本，单链最多串联 2 项变换。命中白名单或低置信度时直接旁路。

### 模块 C：变换还原层（[transforms.py](aedl/transforms.py)）

| 变换 | 操作 | 适用场景 |
|------|------|----------|
| 水平镜像 | `cv2.flip(frame, 1)` | 镜像规避还原 |
| 中心裁剪 | 动态裁剪比例 [0.80, 0.95] | 去除遮挡边框 |
| 亮度增强 | Gamma(γ=0.8) + CLAHE | 还原暗光细节 |
| 轻量去噪 | 高斯 / fastNlMeans | 还原降质画面 |
| 抽帧补偿 | 跳帧采样（零额外计算） | 变速视频 |
| 分屏提取 | 按直线切割取最大子画面 | 多画面拼接 |

### 模块 D：一致性校验层（[consistency.py](aedl/consistency.py)）

对原始与还原版本的七维类别分数向量进行：

1. **单标签跃升**：`Δ = P_restored(label) − P_original(label)`，Δ > 0.4 标记可疑
2. **断崖式跃升**：Δ > 0.5 且 `P_restored > 0.7`，高度疑似
3. **JSD 分布偏移**：Jensen-Shannon 散度 > 0.15 视为显著偏移
4. **多视角投票**：任一还原版本 `P > 0.8` 触发标记
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

- **最终分数**：按标签维度取所有版本保守最大值
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
| 模块 C 变换还原 | 200-500ms / 版本 |
| 后端模型推理 | 取决于原模型（并行后取 max） |
| 模块 D 一致性校验 | < 1ms |
| 模块 E 结果融合 | < 1ms |

仅 10%-15% 视频触发完整流程，整体模型推理量增幅约 30%。

---

## 变更日志

| 日期 | 变更 |
|------|------|
| 2026-06-28 | 初版：实现方案一完整五段式架构，含 A/B/C/D/E 五模块与 FastAPI 服务入口 |
