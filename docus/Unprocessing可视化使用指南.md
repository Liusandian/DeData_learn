# Unprocessing Pipeline 可视化使用指南

> **目标**：通过可视化每个逆ISP步骤，深入理解数据退化的各个环节

---

## 📋 功能概述

已为 `unprocessing.py` 的每个处理步骤添加了**可视化对比功能**，可以清晰地看到：
- 左图：处理前的图像
- 右图：处理后的图像
- 统计信息：Min、Max、Mean值

### 可视化输出内容

| 文件名 | 内容 | 说明 |
|--------|------|------|
| `step1_inverse_gamma.png` | 逆Gamma校正 | sRGB → 线性RGB |
| `step2_inverse_tone_mapping.png` | 逆色调映射 | 扩展动态范围（LDR→HDR） |
| `step3_inverse_color_correction.png` | 逆色彩校正 | sRGB色彩空间 → 相机原生色彩空间 |
| `step4_inverse_white_balance.png` | 逆白平衡 | 移除色温校正，恢复原始色调 |
| `step5_mosaic.png` | Mosaic马赛克化 | RGB三通道 → Bayer单通道 |
| `step6_add_noise.png` | 添加噪声 | 干净Bayer → 带噪声Bayer |
| `step6_noise_detail.png` | 噪声详细分析 | 噪声分布、直方图 |
| `pipeline_summary.png` | **完整流程总结** | 所有步骤的可视化汇总 |

---

## 🚀 快速开始

### 1. 基本用法（带可视化）

```bash
python unprocessing.py \
    --input test_image.jpg \
    --output raw_output.png \
    --iso 1600 \
    --visualize \
    --vis-dir visualization
```

**参数说明：**
- `--input`: 输入的sRGB图像路径
- `--output`: 输出的Raw Bayer图像路径
- `--iso`: ISO值（100, 400, 800, 1600, 3200, 6400）
- `--visualize`: **启用可视化功能**（新增）
- `--vis-dir`: 可视化结果保存目录（默认：`visualization`）

### 2. 处理结果

运行后会生成：
```
visualization/
├── step1_inverse_gamma.png           # 步骤1：逆Gamma
├── step2_inverse_tone_mapping.png    # 步骤2：逆色调映射
├── step3_inverse_color_correction.png # 步骤3：逆色彩校正
├── step4_inverse_white_balance.png   # 步骤4：逆白平衡
├── step5_mosaic.png                  # 步骤5：Mosaic
├── step6_add_noise.png               # 步骤6：添加噪声
├── step6_noise_detail.png            # 步骤6：噪声详细分析
└── pipeline_summary.png              # 完整流程总结
```

---

## 📊 各步骤详解

### 步骤 1: 逆 Gamma 校正

**目的**：从 sRGB 非线性空间转回线性空间

**可视化内容**：
- 左图：sRGB 图像（Gamma编码后，用于显示）
- 右图：线性 RGB（物理光强的线性表示）

**关键观察点**：
- 线性RGB看起来会**更暗**（因为Gamma 2.4的解码）
- 高光区域变化更明显
- 阴影区域差异较小

**物理意义**：
```
sRGB (for display) → Linear RGB (physical light intensity)
```

---

### 步骤 2: 逆色调映射

**目的**：扩展动态范围，模拟Raw的高动态范围

**可视化内容**：
- 左图：线性 RGB（LDR，8-bit范围）
- 右图：扩展后的线性 RGB（HDR，模拟12-14bit Raw）

**关键观察点**：
- 整体亮度提升
- 高光区域得到更多"呼吸空间"
- 缩放因子通常在 1.0 - 1.5 之间

**技术细节**：
```python
scale = 0.95 / percentile_99(image)
scaled_image = image * scale
```

---

### 步骤 3: 逆色彩校正 (CCM)

**目的**：从 sRGB 标准色彩空间转到相机原生色彩空间

**可视化内容**：
- 左图：sRGB 色彩空间（标准化、设备无关）
- 右图：相机原生色彩空间（设备相关）

**关键观察点**：
- **颜色会发生偏移**（这是正常的！）
- 红色/蓝色通道可能出现明显变化
- 绿色相对稳定（绿色通道对亮度贡献大）

**色彩矩阵**：
```
Camera_RGB = CCM^(-1) × sRGB
```

论文提供的平均逆CCM：
```
[ 1.0234, -0.2969, -0.2266]
[-0.5625,  1.6328, -0.0469]
[-0.0703, -0.2188,  1.2891]
```

---

### 步骤 4: 逆白平衡

**目的**：移除白平衡校正，恢复原始色温

**可视化内容**：
- 左图：白平衡后的图像（色温已校正为中性）
- 右图：白平衡前的图像（保留原始色温，可能偏暖/偏冷）

**关键观察点**：
- 图像可能出现**明显色偏**（偏黄/偏蓝）
- 这是原始场景的真实色温
- Red Gain 通常 > Blue Gain

**增益范围**（统计自真实相机数据）：
```
Red Gain:   1.9 - 2.4
Green Gain: 1.0 (固定基准)
Blue Gain:  1.5 - 1.9
```

**公式**：
```python
R_original = R_balanced / red_gain
G_original = G_balanced / green_gain
B_original = B_balanced / blue_gain
```

---

### 步骤 5: Mosaic（马赛克化）

**目的**：从 RGB 三通道转回 Bayer 单通道

**可视化内容**：
- 左图：RGB 三通道图像（每个像素有R、G、B三个值）
- 右图：Bayer 单通道图像（每个像素只保留一个颜色）

**Bayer Pattern (RGGB)**：
```
R  G  R  G
G  B  G  B
R  G  R  G
G  B  G  B
```

**关键观察点**：
- 右图是灰度图，但**包含了所有颜色信息**（以空间分布方式）
- 可以看到明显的"马赛克"纹理（放大后）
- 绿色像素占50%（因为人眼对绿色最敏感）

---

### 步骤 6: 添加噪声

**目的**：模拟相机传感器的噪声（泊松-高斯组合）

**可视化内容**：
- **主图**：干净Bayer vs 噪声Bayer
- **详细分析图**：
  1. 干净图像
  2. 噪声图像
  3. 噪声分布（彩色热图）
  4. 噪声直方图

**噪声模型**：
```
噪声 = 泊松噪声（光子噪声） + 高斯噪声（读取噪声）
```

**关键观察点**：
- ISO越高，噪声越明显
- 亮部噪声 > 暗部噪声（泊松特性）
- 噪声直方图应接近高斯分布

**ISO对应的噪声水平**：
| ISO | 噪声标准差 (典型值) | 视觉效果 |
|-----|-------------------|---------|
| 100-400 | 0.001 - 0.003 | 几乎不可见 |
| 800 | 0.005 - 0.008 | 轻微噪声 |
| 1600 | 0.010 - 0.015 | 明显噪声 |
| 3200+ | 0.020+ | 严重噪声 |

---

## 🎯 Pipeline Summary（流程总结图）

**`pipeline_summary.png`** 是整个流程的**一站式总览**：

### 布局结构
```
┌────────────┬────────────┬────────────┐
│ 步骤0      │ 步骤1      │ 步骤2      │
│ 输入sRGB   │ 线性RGB    │ 扩展范围   │
│            │            │            │
├────────────┼────────────┼────────────┤
│ 步骤3      │ 步骤4      │ 步骤5      │
│ 相机RGB    │ 逆白平衡   │ Bayer      │
│            │            │            │
├────────────┼────────────┼────────────┤
│ 步骤6      │ Bayer示意  │ 参数汇总   │
│ 加噪声     │            │            │
└────────────┴────────────┴────────────┘
```

### 包含信息
- ✅ 所有7个步骤的可视化
- ✅ 步骤间的箭头流向
- ✅ Bayer Pattern 彩色示意图
- ✅ 关键参数汇总（CCM、白平衡、ISO等）
- ✅ 输入输出形状信息

---

## 💡 使用场景

### 1. 学习逆ISP流程
```bash
# 使用一张标准测试图
python unprocessing.py \
    --input test_images/colorchecker.jpg \
    --output raw.png \
    --iso 800 \
    --visualize
```

**然后观察**：
1. `step1_*.png` - 理解Gamma校正的作用
2. `step4_*.png` - 观察白平衡如何改变色温
3. `step6_*.png` - 分析噪声分布特性

### 2. 调试数据集生成
```bash
# 测试不同ISO的噪声水平
for iso in 400 800 1600 3200; do
    python unprocessing.py \
        --input image.jpg \
        --output raw_iso${iso}.png \
        --iso $iso \
        --visualize \
        --vis-dir vis_iso${iso}
done
```

### 3. 对比不同Bayer模式
```bash
# RGGB vs BGGR
python unprocessing.py --input img.jpg --bayer-pattern RGGB --visualize --vis-dir vis_rggb
python unprocessing.py --input img.jpg --bayer-pattern BGGR --visualize --vis-dir vis_bggr
```

---

## 🔍 关键观察点汇总

### 理解数据退化的6个要点

| 步骤 | 现象 | 原因 | 是否正常 |
|-----|------|------|---------|
| 1. 逆Gamma | 图像变暗 | Gamma 2.4 解码 | ✅ 正常 |
| 2. 逆Tone Mapping | 整体变亮 | 扩展动态范围 | ✅ 正常 |
| 3. 逆CCM | 颜色偏移 | 色彩空间转换 | ✅ 正常 |
| 4. 逆白平衡 | 明显色偏 | 恢复原始色温 | ✅ 正常 |
| 5. Mosaic | 变为灰度 | RGB→单通道 | ✅ 正常 |
| 6. 加噪声 | 出现噪点 | 传感器噪声 | ✅ 正常 |

### 异常情况检测

❌ **需要注意的异常**：
- Min/Max 超出 [0, 1] 范围（应该被clip）
- 步骤3、4后出现**大片黑色/白色**（可能CCM/增益不合理）
- 噪声标准差异常大（检查ISO设置）
- Bayer图完全黑/白（检查前面步骤）

---

## 📁 代码改动说明

### 新增功能

1. **初始化参数**：
```python
UnprocessingPipeline(
    visualize=True,        # 是否启用可视化
    output_dir='vis'       # 可视化保存目录
)
```

2. **可视化函数**：
```python
_visualize_comparison()      # 通用对比可视化
_visualize_noise_detail()    # 噪声详细分析
_visualize_pipeline_summary() # 流程总结
_create_bayer_pattern_demo() # Bayer示意图
```

3. **自动调用**：
- 每个步骤完成后自动保存可视化
- 最后生成Pipeline总结图

### 命令行新增参数

```bash
--visualize              # 启用可视化（默认False）
--vis-dir <path>         # 可视化目录（默认'visualization'）
```

---

## 🎓 进阶使用：在代码中调用

### Python脚本调用

```python
from unprocessing import UnprocessingPipeline
import cv2

# 读取图像
img = cv2.imread('test.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 创建Pipeline（启用可视化）
unprocessor = UnprocessingPipeline(
    random_ccm=True,
    random_gains=True,
    add_noise=True,
    visualize=True,           # 启用可视化
    output_dir='my_vis'       # 自定义目录
)

# 处理（会自动保存所有可视化）
raw, metadata = unprocessor.unprocess(img, iso=1600, verbose=True)

# 查看可视化结果
# my_vis/step1_inverse_gamma.png
# my_vis/step2_inverse_tone_mapping.png
# ...
# my_vis/pipeline_summary.png
```

### 批量处理示例

```python
import os
from glob import glob

# 创建Pipeline
unprocessor = UnprocessingPipeline(visualize=True, output_dir='batch_vis')

# 批量处理
for img_path in glob('images/*.jpg'):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 每张图像都会生成可视化
    raw, meta = unprocessor.unprocess(img, iso=800, verbose=False)
    
    # 可视化会自动覆盖（只保留最后一张）
    # 如需保留每张图的可视化，可以动态修改output_dir
```

---

## 📚 相关文档

- **论文原文**：`Unprocessing-Images论文详解.md`
- **代码说明**：`Unprocessing代码使用指南.md`
- **ISP详解**：`AI-ISP数据退化详解.md`

---

## ❓ 常见问题

### Q1: 可视化图片太多，能只生成某个步骤吗？
**A**: 目前是全流程可视化。如需单步骤，可以修改代码注释掉其他步骤的 `_visualize_comparison()` 调用。

### Q2: 可视化图片分辨率太低？
**A**: 修改保存时的 `dpi` 参数：
```python
plt.savefig(save_path, dpi=300, bbox_inches='tight')  # 默认150
```

### Q3: 为什么步骤3、4后颜色看起来很奇怪？
**A**: 这是**正常现象**！逆CCM和逆白平衡会破坏颜色平衡，因为Raw数据本就是"未处理"的原始状态。

### Q4: Bayer图为什么是灰度的？
**A**: Bayer是**单通道**图像，每个像素只有一个颜色值。显示为灰度是为了观察亮度分布，实际上它包含RGB三色信息（以空间分布方式）。

---

## 🎯 总结

通过可视化，你可以：
1. ✅ **直观理解**每个逆ISP步骤的作用
2. ✅ **观察数据退化**的渐进过程
3. ✅ **调试问题**（检查异常值、颜色偏移）
4. ✅ **学习ISP原理**（正向ISP是逆过程）
5. ✅ **验证实现**（对比论文描述）

**推荐工作流**：
```
输入测试图 → 运行带可视化 → 逐步查看每个PNG → 理解变化 → 调整参数 → 重复
```

**关键文件**：
- `pipeline_summary.png` - **必看**，全流程概览
- `step4_*.png` - 理解白平衡
- `step6_noise_detail.png` - 理解噪声模型

---

*最后更新：2026-02-08*

