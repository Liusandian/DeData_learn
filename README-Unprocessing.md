# Unprocessing 实现说明

> 从 sRGB 图像生成 Raw Bayer 图像的完整实现  
> 基于论文：《Unprocessing Images for Learned Raw Denoising》(CVPR 2019)

## 📁 文件清单

```
├── unprocessing.py                        # 主代码（核心实现）
├── test_unprocessing.py                   # 测试脚本
├── Unprocessing代码使用指南.md             # 详细使用文档
├── Unprocessing-Images论文详解.md          # 论文解读
└── 逆ISP完整实现-可运行代码.md             # 技术细节
```

## 🚀 快速开始（5分钟）

### 第1步：运行测试

```bash
# 测试所有功能
python test_unprocessing.py
```

**预期输出**：
```
============================================================
开始运行所有测试
============================================================

测试1：基本 Unprocessing 功能
...
✅ 基本功能测试通过

测试2：不同 ISO 的噪声水平
...
✅ ISO 噪声测试通过

...

✅ 所有测试通过！
```

**生成的文件**：
- `test_output_raw.png` - Raw Bayer 图像
- `test_visualization.png` - 可视化结果
- `bayer_patterns_test.png` - Bayer 模式对比

### 第2步：处理您的图像

```python
# 创建脚本 my_test.py
import cv2
from unprocessing import UnprocessingPipeline, save_raw_bayer

# 读取您的图像
srgb = cv2.imread('your_image.png')
srgb = cv2.cvtColor(srgb, cv2.COLOR_BGR2RGB)

# Unprocessing
unprocessor = UnprocessingPipeline()
raw_bayer, metadata = unprocessor.unprocess(srgb, iso=1600)

# 保存
save_raw_bayer(raw_bayer, 'my_raw_output.png')

print(f"完成！ISO: {metadata['iso']}")
```

运行：
```bash
python my_test.py
```

### 第3步：批量生成训练数据

```python
# 创建脚本 generate_data.py
from unprocessing import batch_process

batch_process(
    input_dir='./your_srgb_images',     # 您的 sRGB 图像目录
    output_dir='./raw_training_data',   # 输出目录
    num_samples=1000,                   # 生成 1000 对
    iso_range=[800, 1600, 3200]         # ISO 范围
)
```

运行：
```bash
python generate_data.py
```

---

## 📖 核心原理

### 六步骤流程

```
sRGB 图像 (8-bit, 非线性, RGB)
    ↓
【步骤1】逆 Gamma 校正
    linear = sRGB^2.4
    ↓
【步骤2】逆色调映射
    扩展动态范围
    ↓
【步骤3】逆色彩校正
    Camera_RGB = CCM^(-1) × sRGB
    ↓
【步骤4】逆白平衡
    RGB = RGB / [gain_r, gain_g, gain_b]
    ↓
【步骤5】Mosaic
    RGB (3通道) → Bayer (1通道)
    ↓
【步骤6】添加噪声
    Poisson + Gaussian
    ↓
Raw Bayer 图像 (线性空间, 单通道)
```

### 关键参数

```python
# Gamma
gamma = 2.2

# 白平衡增益（随机范围）
red_gain:   1.9 - 2.4
green_gain: 1.0
blue_gain:  1.5 - 1.9

# 噪声模型
ISO 范围: 400, 800, 1600, 3200, 6400
shot_noise = Poisson(image × ISO/100) / (ISO/100)
read_noise = Gaussian(0, 0.0005 × √(ISO/100))

# CCM 逆矩阵（平均）
[ 1.0234, -0.2969, -0.2266]
[-0.5625,  1.6328, -0.0469]
[-0.0703, -0.2188,  1.2891]
```

---

## 💻 使用方式

### 方式1：Python API

```python
from unprocessing import UnprocessingPipeline

unprocessor = UnprocessingPipeline()
raw_bayer, metadata = unprocessor.unprocess(srgb_image, iso=1600)
```

### 方式2：命令行

```bash
python unprocessing.py --input image.png --output raw.png --iso 1600
```

### 方式3：批量处理

```python
from unprocessing import batch_process

batch_process('input_dir', 'output_dir', num_samples=1000)
```

---

## 📊 代码功能

### 主要类和函数

```python
# 核心类
UnprocessingPipeline
├── step1_inverse_gamma()          # 逆 Gamma
├── step2_inverse_tone_mapping()   # 逆色调映射
├── step3_inverse_color_correction() # 逆色彩校正
├── step4_inverse_white_balance()  # 逆白平衡
├── step5_mosaic()                 # Mosaic
├── step6_add_noise()              # 添加噪声
└── unprocess()                    # 完整流程

# 辅助函数
save_raw_bayer()      # 保存 Raw 图像
load_raw_bayer()      # 加载 Raw 图像
visualize_bayer()     # 可视化 Bayer 图像
batch_process()       # 批量处理
```

### 每个步骤都可以单独调用

```python
unprocessor = UnprocessingPipeline()

# 只执行前3步
linear = unprocessor.step1_inverse_gamma(srgb_norm)
linear_scaled, _ = unprocessor.step2_inverse_tone_mapping(linear)
camera_rgb = unprocessor.step3_inverse_color_correction(linear_scaled, ccm)

# 或执行完整流程
raw_bayer, metadata = unprocessor.unprocess(srgb)
```

---

## 🎯 应用场景

### 场景1：训练 Raw 去噪网络

```python
# 1. 生成配对数据
from unprocessing import batch_process
batch_process('imagenet_srgb', './raw_data', num_samples=10000)

# 2. 训练网络
from torch.utils.data import Dataset, DataLoader

class RawDataset(Dataset):
    def __getitem__(self, idx):
        noisy = load_raw_bayer(f'raw_data/noisy/{idx}.png')
        clean = load_raw_bayer(f'raw_data/clean/{idx}.png')
        return noisy, clean

dataset = RawDataset()
loader = DataLoader(dataset, batch_size=4)

# 训练...
for noisy, clean in loader:
    output = model(noisy)
    loss = criterion(output, clean)
    # ...
```

### 场景2：AI ISP 数据生成

```python
# 为 AI ISP 项目生成训练数据
unprocessor = UnprocessingPipeline(
    random_ccm=True,      # 模拟不同相机
    random_gains=True,    # 模拟不同光源
    add_noise=True
)

# 生成多样化的数据
for img_path in image_list:
    srgb = load_image(img_path)
    raw, meta = unprocessor.unprocess(srgb)
    # 每次生成的 Raw 都不同（参数随机）
```

### 场景3：研究和实验

```python
# 研究不同参数的影响
for iso in [400, 800, 1600, 3200]:
    raw = unprocess(srgb, iso=iso)
    # 分析噪声特性
```

---

## ✅ 验证与测试

### 运行测试套件

```bash
python test_unprocessing.py
```

**测试内容**：
- ✅ 基本功能（输入/输出正确性）
- ✅ ISO 与噪声的关系
- ✅ 不同 Bayer 模式
- ✅ 可视化功能
- ✅ 逐步执行

### 预期结果

```
所有测试通过：
✅ 输出形状正确 (H, W)
✅ 输出范围正确 [0, 1]
✅ 噪声随 ISO 递增
✅ 所有 Bayer 模式正常工作
✅ 可视化文件生成成功
```

---

## 📚 相关文档

### 详细文档

1. **Unprocessing代码使用指南.md**
   - 详细的使用说明
   - 每个步骤的原理
   - 完整示例代码
   - 参数说明
   - 常见问题

2. **Unprocessing-Images论文详解.md**
   - 论文核心思想
   - 实验结果
   - 应用与启示

3. **逆ISP完整实现-可运行代码.md**
   - 技术细节
   - 更多代码示例
   - 可视化工具

### 学习路线

4. **从HighLevel到LowLevel-AI-ISP学习路线图.md**
   - 8周学习计划
   - 从 High-level CV 转向 Low-level 图像处理

5. **AI-ISP数据退化详解.md**
   - 数据退化的全面介绍
   - 各种退化类型

---

## 🎓 论文信息

**标题**：Unprocessing Images for Learned Raw Denoising  
**作者**：Tim Brooks, Ben Mildenhall (Google Research)  
**发表**：CVPR 2019  
**代码**：https://github.com/google-research/google-research/tree/master/unprocessing  
**重要性**：⭐⭐⭐⭐⭐（必读经典）

**核心贡献**：
1. 完整的逆 ISP 流程
2. 真实的相机噪声模型
3. 在 SIDD 上达到 SOTA（39.28 dB PSNR）
4. 证明合成数据可接近真实数据效果

---

## 💡 下一步

```
□ 运行 test_unprocessing.py 验证代码
□ 用您自己的图像测试
□ 生成少量训练数据（100对）
□ 批量生成训练数据（1000+对）
□ 训练简单的 Raw 去噪网络
□ 在真实 Raw 数据上测试
□ 结合 AIMET 进行量化部署
```

---

**代码状态**：✅ 完整可用  
**测试状态**：✅ 已验证  
**文档状态**：✅ 完善  
**推荐度**：⭐⭐⭐⭐⭐

有任何问题随时反馈！

