# Pocket4 Sensor 线性噪声模型标定 — 使用指南

## 1. 概述

本工具对 Pocket4 传感器 14-bit Raw 数据执行完整的线性噪声模型标定：

```
σ²(y) = α · μ(y) + β
```

- **α** (DN/e⁻): 系统增益，对应散粒噪声（shot noise），与信号成正比
- **β** (DN²): 读出噪声方差（read noise），信号无关的底噪

标定覆盖 5 增益 × 4 Bayer 通道 = **20 组参数**。

---

## 2. 数据采集要求

### 2.1 采集矩阵

| 维度     | 取值                                            |
|----------|------------------------------------------------|
| 增益档位 | 1 / 2 / 4 / 8 / 1600                           |
| 亮度级   | 黑帧 / 12.5% / 25% / 37.5% / 50% / 62.5% / 75% / 87.5% / 100% |
| 每组帧数 | >= 2 帧（推荐 >= 10 帧，帧数越多时域统计越稳定） |

总计: 5 增益 × 9 亮度级 × N 帧/组

### 2.2 场景设置

- 暗室环境，灯箱提供均匀光源
- 4×6 标准色卡（24色），放置在画面九宫格中心 1/9 区域
- 每个增益下固定曝光，通过灯箱调整亮度

### 2.3 目录结构

```
data_root/
├── gain_1/
│   ├── dark/                  # 黑帧（遮光拍摄）
│   │   ├── frame_000.raw
│   │   ├── frame_001.raw
│   │   └── ...
│   ├── brightness_12.5/       # 亮度 12.5%
│   │   ├── frame_000.raw
│   │   └── ...
│   ├── brightness_25/
│   ├── brightness_37.5/
│   ├── brightness_50/
│   ├── brightness_62.5/
│   ├── brightness_75/
│   ├── brightness_87.5/
│   └── brightness_100/
├── gain_2/
│   └── (同上结构)
├── gain_4/
├── gain_8/
└── gain_1600/
```

支持的 Raw 文件格式: `.raw` (uint16) / `.npy` / `.dng` / `.arw`

---

## 3. 快速上手

### 3.1 安装依赖

```bash
pip install numpy scipy matplotlib
# 可选（Excel导出 / DNG读取）
pip install pandas openpyxl rawpy
```

### 3.2 模拟数据验证

先用模拟数据跑通整个流程，确认环境和代码无问题：

```bash
python pocket4_noise_calibration.py --simulate
```

模拟模式会：
1. 在 `/tmp/sim_pocket4/` 下生成已知参数的模拟 Raw 数据
2. 执行完整标定流程
3. 将标定值与真实值对比，输出误差百分比

### 3.3 真实数据标定

```bash
python pocket4_noise_calibration.py \
    --data_root /path/to/your/raw_data \
    --raw_shape 3024 4032 \
    --bayer RGGB \
    --output_dir ./calibration_results
```

### 3.4 仅用灰阶 patch 拟合

灰阶 patch 光谱平坦、四通道响应一致，标定结果更可靠（数据点更少但更准）：

```bash
python pocket4_noise_calibration.py \
    --data_root /path/to/data \
    --raw_shape 3024 4032 \
    --gray_only
```

---

## 4. 标定流程详解

### 4.1 流程总览

```
Raw 数据
  │
  ▼
Step 0: 数据目录扫描 ─────────── 自动识别目录命名
  │
  ▼
Step 1: BLC 逐通道标定 ────────── 从黑帧提取 R/Gr/Gb/B 各自的黑电平
  │
  ▼
Step 2: ROI 选取 ──────────────── 定位 24 个 patch，取中心 60% 区域
  │
  ▼
Step 3: 均值-方差计算 ─────────── 时域统计（>=3帧）或帧差法（2帧）
  │
  ▼
Step 4: 线性回归拟合 ──────────── σ² = α·μ + β，含离群点剔除
  │
  ▼
Step 5: 参数验证 ──────────────── R²/增益比例/黑帧交叉验证
  │
  ▼
Step 6: 可视化 & 导出 ─────────── PTC 曲线 + JSON + Excel
```

### 4.2 BLC 标定的关键：先拆通道，再逐通道减

```python
# 错误做法：在 Bayer 拆分前减去全局 BLC
raw = raw - global_blc          # ✗ 四个通道 BLC 不同，这会引入偏差

# 正确做法：先拆通道，再逐通道减各自的 BLC
R  = raw[0::2, 0::2]  →  R  -= BLC_R
Gr = raw[0::2, 1::2]  →  Gr -= BLC_Gr
Gb = raw[1::2, 0::2]  →  Gb -= BLC_Gb
B  = raw[1::2, 1::2]  →  B  -= BLC_B
```

**为什么**：R/Gr/Gb/B 四个通道对应不同的读出电路，暗电流和偏置电压有差异，
黑电平差异可达数个甚至十几个 DN。对暗区低信号的均值影响非常显著。

### 4.3 时域统计 vs 帧差法

| 方法       | 最少帧数 | PRNU 影响 | 精度   | 适用场景       |
|-----------|---------|----------|-------|---------------|
| 时域统计   | >= 3    | 天然消除  | 高    | 帧数充足时首选  |
| 帧差法     | 2       | 天然消除  | 中    | 帧数有限时备选  |
| 空域统计   | 1       | 包含PRNU | 低    | 不推荐用于标定  |

脚本会根据实际帧数自动选择方法。

### 4.4 数据清洗规则

拟合前会自动剔除以下数据点：

1. **饱和点**: 均值（未减 BLC）> 90% 满量程（14-bit 下 > 14745 DN）
2. **负信号**: 减 BLC 后均值 ≤ 0
3. **离群点**: 残差超过 3σ（两轮拟合，第一轮识别离群点，第二轮剔除后重拟合）

Gain=1600 时信号极易饱和，高亮度级数据大概率被剔除，这是预期行为。

---

## 5. 输出文件说明

标定完成后在 `output_dir` 下生成以下文件：

| 文件                      | 内容                                       |
|--------------------------|--------------------------------------------|
| `pocket4_noise_model.json` | 完整标定参数（α, β, BLC, R², 读出噪声, 满阱, 动态范围） |
| `pocket4_noise_model.xlsx` | Excel 格式参数表（需 pandas）              |
| `ptc_curves.png`          | PTC 曲线（每个增益一个子图，四通道分色绘制）|
| `noise_model_summary.png` | α/β 随增益变化趋势图                       |
| `validation_report.txt`   | 验证报告（R²/比例/交叉验证结果）           |

### 5.1 JSON 输出示例

```json
{
  "sensor_model": "Pocket4",
  "bit_depth": 14,
  "bayer_pattern": "RGGB",
  "noise_model": "linear: var = alpha * mu + beta",
  "params": {
    "gain_1": {
      "R": {
        "alpha": 0.50123456,
        "beta": 2.0134,
        "black_level": 1024.50,
        "r_squared": 0.999812,
        "read_noise_e": 2.8321,
        "full_well_capacity_e": 30653.2,
        "dynamic_range_dB": 80.69
      }
    }
  }
}
```

---

## 6. 验证标准

### 6.1 拟合质量

| R² 范围     | 评价 | 行动                                 |
|------------|------|--------------------------------------|
| > 0.999    | 优秀 | 无需额外操作                          |
| 0.99~0.999 | 合格 | 可用，建议检查是否有少量异常点未剔除   |
| 0.95~0.99  | 警告 | 需排查数据（饱和未剔除？BLC 不准？ROI 不均匀？） |
| < 0.95     | 不合格 | 线性模型假设不成立或数据有严重问题    |

### 6.2 增益比例关系

对模拟增益，以下关系应成立：

```
α(Gain=g) / α(Gain=1) ≈ g          （α 与增益成正比）
β(Gain=g) / β(Gain=1) ≈ g²         （β 与增益平方成正比）
```

偏差 > 20% 时需排查：
- 是否为数字增益（数字增益下 β 与 g² 成正比但 α 也与 g² 成正比）
- Gain=1600 是否为混合增益模式

### 6.3 交叉验证

拟合截距 β 应与黑帧时域方差接近，偏差 < 20%。
偏差大的可能原因：暗室漏光、黑帧帧数不足、BLC 计算有误。

---

## 7. 标定参数的使用

### 7.1 noiseMap 生成

```python
# 在 ISP 降噪模块中，根据标定参数生成逐像素 noiseMap
def generate_noise_map(raw_channel, alpha, beta, blc):
    """
    raw_channel: 单通道 Raw 数据 (DN)
    alpha, beta: 标定参数
    blc: 黑电平
    返回: noise_std_map, 逐像素噪声标准差 (DN)
    """
    signal = np.maximum(raw_channel - blc, 0)  # 净信号
    variance = alpha * signal + beta            # 逐像素方差
    variance = np.maximum(variance, beta)       # 方差不低于读出噪声
    return np.sqrt(variance)                    # 标准差
```

### 7.2 降噪网络输入

noiseMap 作为 NAFNet 等降噪网络的额外输入通道：

```python
# 拼接带噪图像和 noiseMap
input_tensor = torch.cat([noisy_raw, noise_map], dim=1)  # (B, C+1, H, W)
denoised = nafnet(input_tensor)
```

### 7.3 噪声模拟（训练数据合成）

```python
# 用标定参数合成训练数据中的噪声
def add_calibrated_noise(clean_raw, alpha, beta, blc):
    signal = np.maximum(clean_raw - blc, 0)
    std_map = np.sqrt(alpha * signal + beta)
    noise = np.random.randn(*clean_raw.shape) * std_map
    return clean_raw + noise
```

---

## 8. 脚本函数索引

| 函数名 | 位置 | 功能 |
|--------|------|------|
| `load_single_raw()` | Raw 加载 | 加载单帧 Raw（.raw/.npy/.dng） |
| `load_multi_frames()` | Raw 加载 | 批量加载多帧 Raw |
| `split_bayer_channels()` | Bayer 拆分 | 单帧 → R/Gr/Gb/B 四通道 |
| `split_bayer_multi_frames()` | Bayer 拆分 | 多帧批量拆分 |
| `calibrate_blc_per_channel()` | BLC 标定 | 黑帧 → 逐通道黑电平 + 暗帧方差 |
| `compute_patch_rois()` | ROI 选取 | 色卡 24 patch 的 ROI 坐标计算 |
| `compute_mean_variance_temporal()` | 均值方差 | 时域统计（>=3帧） |
| `compute_mean_variance_frame_diff()` | 均值方差 | 帧差法（2帧） |
| `fit_linear_noise_model()` | 线性拟合 | WLS 回归 + 离群点剔除 |
| `validate_calibration()` | 参数验证 | R²/比例/交叉验证报告 |
| `plot_ptc_curves()` | 可视化 | PTC 曲线绘制 |
| `plot_noise_model_summary()` | 可视化 | α/β 趋势图 |
| `export_results_json()` | 导出 | JSON 格式标定结果 |
| `export_results_excel()` | 导出 | Excel 格式参数表 |
| `generate_simulated_data()` | 模拟 | 生成已知参数的模拟 Raw 数据 |
| `run_calibration()` | 主流程 | 完整标定 6 步流程 |
| `run_simulation_validation()` | 验证 | 模拟数据标定 + 精度对比 |

---

## 9. 常见问题

### Q: Gain=1600 只拟合了很少的数据点？

正常现象。Gain=1600 增益极高，信号很容易饱和。高亮度级（如 75%/87.5%/100%）
的数据大概率超过饱和阈值被自动剔除。有效数据可能只有低亮度的几级。

### Q: R² 偏低怎么排查？

按优先级排查：
1. 是否有饱和数据点未被正确剔除
2. BLC 是否准确（检查黑帧是否有漏光）
3. ROI 区域是否均匀（patch 内有反光、渐变）
4. 高信号区是否偏离线性（考虑非线性噪模）

### Q: 报错 "时域统计至少需要 2 帧"？

每组条件下至少需要采集 2 帧 Raw。建议采集 10+ 帧以获得稳定的时域统计。

### Q: 如何修改色卡在画面中的位置？

修改 `compute_patch_rois()` 中的色卡区域边界：
```python
# 当前: 中心 1/9 区域
chart_y1 = sub_h // 3
chart_y2 = 2 * sub_h // 3
# 改为其他位置，按实际坐标修改即可
```
