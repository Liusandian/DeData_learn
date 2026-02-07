# 从 High-level CV 到 AI ISP：8年经验者的转型学习路线图

> **目标受众**：有8年 High-level CV 经验，转向 AI ISP 和 Low-level 图像处理  
> **学习周期**：6-8周（全职学习）或 3-4个月（业余学习）  
> **难度等级**：⭐⭐⭐（有深度学习基础，主要是领域知识补充）

## 目录
- [0. 您的优势与知识缺口分析](#0-您的优势与知识缺口分析)
- [1. 第一阶段：图像质量评价指标（Week 1）](#1-第一阶段图像质量评价指标week-1)
- [2. 第二阶段：ISP 流程与原理（Week 2）](#2-第二阶段isp-流程与原理week-2)
- [3. 第三阶段：图像退化模型（Week 3-4）](#3-第三阶段图像退化模型week-3-4)
- [4. 第四阶段：感知损失与高级技术（Week 5）](#4-第四阶段感知损失与高级技术week-5)
- [5. 第五阶段：实战项目（Week 6-8）](#5-第五阶段实战项目week-6-8)
- [6. 进阶路线：成为专家](#6-进阶路线成为专家)
- [7. 资源汇总](#7-资源汇总)

---

## 0. 您的优势与知识缺口分析

### 0.1 您已有的优势（可直接复用）

```python
✅ PyTorch 深度学习基础
   → 网络设计、训练、调试技巧
   → 可直接应用于 Low-level 任务

✅ 模型训练和调优经验
   → 学习率调度、正则化、数据增强
   → 迁移到图像复原任务

✅ 数据处理技巧
   → DataLoader、数据预处理
   → 改造为退化数据生成管道

✅ GPU 优化经验
   → 混合精度训练、显存优化
   → Low-level 任务计算量更大，更需要优化

✅ 部署经验
   → 模型量化、剪枝、导出
   → AI ISP 需要在移动端/边缘设备部署
```

### 0.2 知识缺口（需要补充）

```python
📚 图像退化模型
   → 需要理解：噪声、模糊、下采样等退化如何建模
   → 为什么重要：生成训练数据的核心
   → 学习周期：2周

📚 ISP 流程知识
   → 需要理解：从 Raw 到 RGB 的完整流程
   → 为什么重要：AI ISP 是对传统 ISP 的替代/增强
   → 学习周期：1周

📚 PSNR/SSIM 等指标
   → 需要理解：如何评价图像质量
   → 为什么重要：Loss 函数设计、模型评估
   → 学习周期：3天

📚 感知损失理解
   → 需要理解：基于深度特征的损失函数
   → 为什么重要：提升复原图像的感知质量
   → 学习周期：1周
```

### 0.3 学习策略

```
原则1：80/20 法则
  - 80% 时间实践，20% 时间理论
  - 您已有深度学习基础，重点是领域知识

原则2：快速迭代
  - 每个阶段都有实践项目
  - 边学边做，快速验证理解

原则3：论文驱动
  - 通过复现经典论文快速掌握技术
  - 看代码比看教材更快（对您而言）

原则4：对比学习
  - High-level vs Low-level 对比
  - 利用已有经验加速理解
```

---

## 1. 第一阶段：图像质量评价指标（Week 1）

> **目标**：掌握 PSNR、SSIM、LPIPS 等评价指标，理解其原理和局限性  
> **时间**：5-7天  
> **难度**：⭐⭐

### 1.1 为什么从这里开始？

```
High-level 任务评价：
├─ 分类：Accuracy, Top-5
├─ 检测：mAP, IoU
├─ 分割：mIoU, Dice
└─ 深度估计：绝对误差、相对误差

Low-level 任务评价：
├─ PSNR（峰值信噪比）         ← 新知识
├─ SSIM（结构相似性）         ← 新知识
├─ LPIPS（感知相似性）        ← 新知识
└─ FID（Frechet Inception Distance）

为什么先学这个？
✅ 基础且重要（所有论文都用）
✅ 相对独立（不依赖其他知识）
✅ 快速上手（3天即可掌握）
✅ 后续实践都需要用到
```

### 1.2 学习内容

#### Day 1-2: PSNR（峰值信噪比）

**理论**：

```python
# PSNR 定义
PSNR = 10 * log10(MAX^2 / MSE)

其中：
- MAX: 像素最大值（通常是 255）
- MSE: 均方误差 = mean((img1 - img2)^2)

物理意义：
- 衡量重建图像与原图的像素级差异
- 单位：dB（分贝）
- 典型范围：20-40 dB
  - <20 dB: 质量很差
  - 25-30 dB: 可接受
  - 30-35 dB: 良好
  - >35 dB: 优秀

优点：
✅ 计算简单
✅ 数学性质好（可微分）
✅ 广泛使用

缺点：
❌ 不完全符合人眼感知
❌ 对平移、旋转敏感
❌ 只关注像素级差异
```

**代码实现**：

```python
import numpy as np
import cv2

def calculate_psnr(img1, img2):
    """
    计算 PSNR
    
    Args:
        img1, img2: numpy array, shape (H, W, C), range [0, 255]
    
    Returns:
        psnr: float, 单位 dB
    """
    # 转为浮点数
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # 计算 MSE
    mse = np.mean((img1 - img2) ** 2)
    
    # 避免除零
    if mse == 0:
        return 100.0  # 完全相同
    
    # 计算 PSNR
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    return psnr

# 使用示例
original = cv2.imread('original.png')
restored = cv2.imread('restored.png')
psnr = calculate_psnr(original, restored)
print(f"PSNR: {psnr:.2f} dB")

# PyTorch 版本
import torch

def psnr_torch(img1, img2):
    """
    PyTorch 版本的 PSNR
    
    Args:
        img1, img2: torch.Tensor, shape (B, C, H, W), range [0, 1]
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return torch.tensor(100.0)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))
```

**实践任务**：

```python
# 任务1：实现并测试 PSNR
# 1. 对比不同噪声强度下的 PSNR
clean = cv2.imread('clean.png')
for sigma in [10, 25, 50, 75]:
    noisy = add_gaussian_noise(clean, sigma)
    psnr = calculate_psnr(clean, noisy)
    print(f"Noise sigma={sigma}: PSNR={psnr:.2f} dB")

# 预期结果：
# sigma=10  → PSNR ≈ 28 dB
# sigma=25  → PSNR ≈ 20 dB
# sigma=50  → PSNR ≈ 14 dB
# sigma=75  → PSNR ≈ 10 dB

# 任务2：对比不同压缩质量
for quality in [20, 40, 60, 80, 95]:
    compressed = jpeg_compress(clean, quality)
    psnr = calculate_psnr(clean, compressed)
    print(f"JPEG quality={quality}: PSNR={psnr:.2f} dB")
```

#### Day 3-4: SSIM（结构相似性）

**理论**：

```python
# SSIM 定义
SSIM(x, y) = [l(x,y)]^α · [c(x,y)]^β · [s(x,y)]^γ

其中：
- l(x,y): 亮度对比
- c(x,y): 对比度对比
- s(x,y): 结构对比

简化版（α=β=γ=1）：
SSIM = (2*μx*μy + C1) * (2*σxy + C2) / 
       ((μx^2 + μy^2 + C1) * (σx^2 + σy^2 + C2))

物理意义：
- 衡量图像的结构相似性
- 范围：[-1, 1]，通常在 [0, 1]
  - 0: 完全不相似
  - 1: 完全相同
- 典型值：
  - <0.8: 质量较差
  - 0.8-0.9: 可接受
  - 0.9-0.95: 良好
  - >0.95: 优秀

优点：
✅ 更符合人眼感知
✅ 考虑结构信息
✅ 对亮度、对比度变化鲁棒

缺点：
❌ 计算复杂（需要窗口滑动）
❌ 参数选择影响结果
```

**代码实现**：

```python
from skimage.metrics import structural_similarity as ssim
import cv2

# 方法1：使用 scikit-image
def calculate_ssim_sklearn(img1, img2):
    """使用 sklearn 计算 SSIM"""
    # 转为灰度图（或使用 multichannel=True 处理彩色图）
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    score = ssim(gray1, gray2, data_range=255)
    return score

# 方法2：彩色图像 SSIM
def calculate_ssim_color(img1, img2):
    """彩色图像 SSIM（对每个通道分别计算再平均）"""
    ssim_channels = []
    for i in range(3):
        score = ssim(img1[:,:,i], img2[:,:,i], data_range=255)
        ssim_channels.append(score)
    return np.mean(ssim_channels)

# 方法3：PyTorch 实现（用于训练时计算）
import torch
import torch.nn.functional as F

def ssim_torch(img1, img2, window_size=11):
    """
    PyTorch 版本的 SSIM
    
    Args:
        img1, img2: torch.Tensor, shape (B, C, H, W), range [0, 1]
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # 创建高斯窗口
    def create_window(window_size, channel):
        _1D_window = torch.exp(
            -torch.arange(window_size).float() ** 2 / (2 * (window_size/2) ** 2)
        )
        _1D_window = _1D_window / _1D_window.sum()
        _2D_window = _1D_window.unsqueeze(1) @ _1D_window.unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    channel = img1.size(1)
    window = create_window(window_size, channel).to(img1.device)
    
    # 计算均值
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    # 计算方差和协方差
    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2
    
    # 计算 SSIM
    ssim_map = ((2*mu1_mu2 + C1) * (2*sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean()

# 使用示例
original = cv2.imread('original.png')
restored = cv2.imread('restored.png')
ssim_score = calculate_ssim_color(original, restored)
print(f"SSIM: {ssim_score:.4f}")
```

**实践任务**：

```python
# 任务：对比 PSNR 和 SSIM 的差异

import cv2
import numpy as np

clean = cv2.imread('clean.png')

# 退化1：高斯噪声
noisy = add_gaussian_noise(clean, sigma=25)
psnr1 = calculate_psnr(clean, noisy)
ssim1 = calculate_ssim_color(clean, noisy)

# 退化2：高斯模糊
blurred = cv2.GaussianBlur(clean, (15, 15), 5)
psnr2 = calculate_psnr(clean, blurred)
ssim2 = calculate_ssim_color(clean, blurred)

# 退化3：JPEG 压缩
compressed = jpeg_compress(clean, quality=30)
psnr3 = calculate_psnr(clean, compressed)
ssim3 = calculate_ssim_color(clean, compressed)

print(f"{'退化类型':<10} {'PSNR':<10} {'SSIM':<10}")
print(f"{'噪声':<10} {psnr1:.2f} dB  {ssim1:.4f}")
print(f"{'模糊':<10} {psnr2:.2f} dB  {ssim2:.4f}")
print(f"{'压缩':<10} {psnr3:.2f} dB  {ssim3:.4f}")

# 观察：
# - 模糊可能 PSNR 高但 SSIM 低（结构破坏严重）
# - 噪声可能 PSNR 低但 SSIM 中等（结构保留）
# 这说明 SSIM 更关注结构信息
```

#### Day 5: LPIPS（感知相似性）

**理论**：

```python
# LPIPS（Learned Perceptual Image Patch Similarity）

核心思想：
使用预训练的深度网络（如 VGG）提取特征，
在特征空间计算距离，而不是像素空间

LPIPS(x, y) = Σ wl * ||Fl(x) - Fl(y)||^2

其中：
- Fl: 第 l 层的特征
- wl: 该层的权重（通过学习得到）

物理意义：
- 衡量感知上的相似性
- 范围：[0, +∞)，通常 [0, 1]
  - 0: 感知上完全相同
  - <0.1: 非常相似
  - 0.1-0.3: 中等相似
  - >0.3: 差异明显

与 PSNR/SSIM 的区别：
PSNR/SSIM → 像素级/结构级相似性
LPIPS     → 感知级相似性（更接近人眼）
```

**代码实现**：

```python
# 安装 LPIPS
# pip install lpips

import lpips
import torch
from PIL import Image
import torchvision.transforms as transforms

# 初始化 LPIPS 模型
loss_fn_alex = lpips.LPIPS(net='alex')  # 使用 AlexNet
loss_fn_vgg = lpips.LPIPS(net='vgg')    # 使用 VGG

def calculate_lpips(img1_path, img2_path):
    """计算两张图像的 LPIPS 距离"""
    # 加载图像
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                           std=[0.5, 0.5, 0.5])
    ])
    
    img1 = transform(Image.open(img1_path)).unsqueeze(0)
    img2 = transform(Image.open(img2_path)).unsqueeze(0)
    
    # 计算 LPIPS
    with torch.no_grad():
        lpips_alex = loss_fn_alex(img1, img2)
        lpips_vgg = loss_fn_vgg(img1, img2)
    
    return lpips_alex.item(), lpips_vgg.item()

# 使用
lpips_a, lpips_v = calculate_lpips('original.png', 'restored.png')
print(f"LPIPS (AlexNet): {lpips_a:.4f}")
print(f"LPIPS (VGG): {lpips_v:.4f}")
```

**实践任务**：

```python
# 任务：三种指标的全面对比

def evaluate_all_metrics(clean_path, degraded_path):
    """评估所有指标"""
    # 读取图像
    clean = cv2.imread(clean_path)
    degraded = cv2.imread(degraded_path)
    
    # PSNR
    psnr = calculate_psnr(clean, degraded)
    
    # SSIM
    ssim_score = calculate_ssim_color(clean, degraded)
    
    # LPIPS
    lpips_alex, lpips_vgg = calculate_lpips(clean_path, degraded_path)
    
    print(f"评价指标：")
    print(f"  PSNR:       {psnr:.2f} dB")
    print(f"  SSIM:       {ssim_score:.4f}")
    print(f"  LPIPS(Alex):{lpips_alex:.4f}")
    print(f"  LPIPS(VGG): {lpips_vgg:.4f}")
    
    return {
        'psnr': psnr,
        'ssim': ssim_score,
        'lpips_alex': lpips_alex,
        'lpips_vgg': lpips_vgg
    }

# 测试多种退化
test_cases = [
    ('clean.png', 'noisy_sigma10.png', '轻度噪声'),
    ('clean.png', 'noisy_sigma50.png', '重度噪声'),
    ('clean.png', 'blurred.png', '模糊'),
    ('clean.png', 'compressed_q30.png', '重压缩'),
    ('clean.png', 'compressed_q80.png', '轻压缩'),
]

for clean, degraded, desc in test_cases:
    print(f"\n{desc}:")
    evaluate_all_metrics(clean, degraded)
```

#### Day 6-7: 综合实践与总结

**实践项目：构建评价工具**

```python
"""
综合评价工具：对比不同复原方法
"""

import cv2
import numpy as np
from lpips import LPIPS
import torch
from tabulate import tabulate

class ImageQualityEvaluator:
    """图像质量评价器"""
    def __init__(self):
        self.lpips_fn = LPIPS(net='alex')
    
    def evaluate(self, clean_path, restored_path):
        """全面评价"""
        # 读取图像
        clean = cv2.imread(clean_path)
        restored = cv2.imread(restored_path)
        
        # 计算所有指标
        psnr = self.calculate_psnr(clean, restored)
        ssim = self.calculate_ssim(clean, restored)
        lpips = self.calculate_lpips(clean_path, restored_path)
        
        return {
            'PSNR (dB)': f"{psnr:.2f}",
            'SSIM': f"{ssim:.4f}",
            'LPIPS': f"{lpips:.4f}"
        }
    
    def calculate_psnr(self, img1, img2):
        """计算 PSNR"""
        mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
        if mse == 0:
            return 100.0
        return 20 * np.log10(255.0 / np.sqrt(mse))
    
    def calculate_ssim(self, img1, img2):
        """计算 SSIM（彩色）"""
        from skimage.metrics import structural_similarity as ssim
        ssim_channels = []
        for i in range(3):
            score = ssim(img1[:,:,i], img2[:,:,i], data_range=255)
            ssim_channels.append(score)
        return np.mean(ssim_channels)
    
    def calculate_lpips(self, img1_path, img2_path):
        """计算 LPIPS"""
        from torchvision import transforms
        from PIL import Image
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        
        img1 = transform(Image.open(img1_path)).unsqueeze(0)
        img2 = transform(Image.open(img2_path)).unsqueeze(0)
        
        with torch.no_grad():
            lpips_value = self.lpips_fn(img1, img2)
        
        return lpips_value.item()
    
    def compare_methods(self, clean_path, method_results):
        """
        对比多种方法
        
        Args:
            clean_path: 干净图像路径
            method_results: {方法名: 复原图像路径}
        """
        results = []
        
        for method_name, restored_path in method_results.items():
            metrics = self.evaluate(clean_path, restored_path)
            row = [method_name] + list(metrics.values())
            results.append(row)
        
        # 打印表格
        headers = ['方法', 'PSNR (dB)', 'SSIM', 'LPIPS']
        print(tabulate(results, headers=headers, tablefmt='grid'))

# 使用示例
evaluator = ImageQualityEvaluator()

# 对比不同去噪方法
methods = {
    'Noisy (输入)': 'noisy.png',
    'BM3D': 'bm3d_result.png',
    'DnCNN': 'dncnn_result.png',
    'NAFNet': 'nafnet_result.png',
}

evaluator.compare_methods('clean.png', methods)

# 预期输出：
# ╒═══════════════╤═════════════╤════════╤═══════════╕
# │ 方法          │ PSNR (dB)   │ SSIM   │ LPIPS     │
# ╞═══════════════╪═════════════╪════════╪═══════════╡
# │ Noisy (输入)  │ 20.23       │ 0.6421 │ 0.3245    │
# │ BM3D          │ 28.45       │ 0.8532 │ 0.1234    │
# │ DnCNN         │ 29.87       │ 0.8756 │ 0.0987    │
# │ NAFNet        │ 31.23       │ 0.9012 │ 0.0654    │
# ╘═══════════════╧═════════════╧════════╧═══════════╛
```

### 1.3 阶段总结与检验

**自我检验清单**：

```
□ 能解释 PSNR、SSIM、LPIPS 的物理意义
□ 能手写实现 PSNR 和 SSIM 的计算
□ 理解三种指标的优缺点和适用场景
□ 能使用 PyTorch 实现可微分的指标（用于 Loss）
□ 能搭建完整的评价工具对比不同方法
```

**知识点测试**：

```python
# 测试1：什么情况下 PSNR 高但 SSIM 低？
答案：图像被平移、旋转，或者只改变了亮度，
     像素值差异小（PSNR 高），但结构不同（SSIM 低）

# 测试2：LPIPS 为什么更接近人眼感知？
答案：使用深度特征而非像素值，
     深度网络学到了人类视觉系统的特征表示

# 测试3：训练去噪网络时应该用哪个指标作为 Loss？
答案：常用 L1 或 L2（对应 PSNR）
     如果追求感知质量，可加入 LPIPS 或 Perceptual Loss
```

---

## 2. 第二阶段：ISP 流程与原理（Week 2）

> **目标**：理解从 Raw 到 RGB 的完整 ISP 流程，掌握各模块原理  
> **时间**：5-7天  
> **难度**：⭐⭐⭐

### 2.1 为什么学习 ISP？

```
AI ISP 的本质：
用神经网络替代或增强传统 ISP 的某些模块

传统 ISP:  Raw → [多个模块] → RGB
AI ISP:    Raw → [神经网络] → RGB

理解传统 ISP 的好处：
✅ 知道每个模块在做什么（有的放矢）
✅ 设计更好的网络架构（借鉴传统方法）
✅ 设计更真实的退化模型（逆 ISP）
✅ 理解 AI ISP 的优势在哪里
```

### 2.2 学习内容

#### Day 1: ISP 概览与 Bayer 图像

**核心概念**：

```python
# 相机成像流程
场景光线 → 镜头 → 传感器 (CFA) → Raw 数据 → ISP → RGB 图像

关键概念：
1. Bayer Pattern（拜耳阵列）
   - 传感器上的彩色滤光片排列
   - 典型模式：RGGB
   
   R  G  R  G  R  G
   G  B  G  B  G  B
   R  G  R  G  R  G
   G  B  G  B  G  B
   
   特点：
   - 每个像素只记录一种颜色
   - G 通道数量是 R/B 的2倍（人眼对绿色更敏感）

2. Raw 图像
   - 传感器直接输出的数据
   - 单通道（mosaic 图像）
   - 线性空间（未经 Gamma 校正）
   - 动态范围大（10-14 bit）
```

**代码实践**：

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt

def visualize_bayer_pattern(raw_image):
    """可视化 Bayer 图像"""
    h, w = raw_image.shape
    
    # 分离 RGGB 通道
    r_channel = np.zeros((h//2, w//2))
    g1_channel = np.zeros((h//2, w//2))
    g2_channel = np.zeros((h//2, w//2))
    b_channel = np.zeros((h//2, w//2))
    
    r_channel = raw_image[0::2, 0::2]    # R
    g1_channel = raw_image[0::2, 1::2]   # G (R 行)
    g2_channel = raw_image[1::2, 0::2]   # G (B 行)
    b_channel = raw_image[1::2, 1::2]    # B
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].imshow(r_channel, cmap='Reds')
    axes[0, 0].set_title('R Channel')
    axes[0, 1].imshow((g1_channel + g2_channel) / 2, cmap='Greens')
    axes[0, 1].set_title('G Channel')
    axes[1, 0].imshow(b_channel, cmap='Blues')
    axes[1, 0].set_title('B Channel')
    axes[1, 1].imshow(raw_image, cmap='gray')
    axes[1, 1].set_title('Raw Mosaic')
    
    plt.tight_layout()
    plt.savefig('bayer_visualization.png')

# 实践：从 RGB 生成模拟 Bayer 图像
def rgb_to_bayer(rgb_image):
    """从 RGB 图像生成 Bayer 模式图像（用于学习）"""
    h, w, _ = rgb_image.shape
    bayer = np.zeros((h, w), dtype=np.uint8)
    
    # RGGB 模式
    bayer[0::2, 0::2] = rgb_image[0::2, 0::2, 0]  # R
    bayer[0::2, 1::2] = rgb_image[0::2, 1::2, 1]  # G
    bayer[1::2, 0::2] = rgb_image[1::2, 0::2, 1]  # G
    bayer[1::2, 1::2] = rgb_image[1::2, 1::2, 2]  # B
    
    return bayer

# 测试
rgb = cv2.imread('test.png')
rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
bayer = rgb_to_bayer(rgb)
visualize_bayer_pattern(bayer)
```

#### Day 2: 去马赛克（Demosaicing）

**原理**：

```python
# 去马赛克：从 Bayer 单通道恢复 RGB 三通道

问题：
Bayer 图像中，每个位置只有一个颜色值
需要插值得到其他两个颜色

R  ?  R  ?      R  G  R  G
?  B  ?  B  →   G  B  G  B
R  ?  R  ?      R  G  R  G
?  B  ?  B      G  B  G  B

方法：
1. 双线性插值（Bilinear）- 简单但有伪影
2. 边缘感知插值（Edge-aware）- 更好
3. 深度学习方法（AI Demosaicing）- 最优
```

**代码实现**：

```python
def demosaic_bilinear(bayer):
    """简单的双线性去马赛克"""
    h, w = bayer.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    # R 通道
    r_mask = np.zeros((h, w))
    r_mask[0::2, 0::2] = 1
    rgb[:,:,0] = cv2.resize(
        bayer * r_mask, 
        (w, h), 
        interpolation=cv2.INTER_LINEAR
    )
    
    # G 通道（两个 G 位置的平均）
    g_mask = np.zeros((h, w))
    g_mask[0::2, 1::2] = 1
    g_mask[1::2, 0::2] = 1
    rgb[:,:,1] = cv2.resize(
        bayer * g_mask,
        (w, h),
        interpolation=cv2.INTER_LINEAR
    )
    
    # B 通道
    b_mask = np.zeros((h, w))
    b_mask[1::2, 1::2] = 1
    rgb[:,:,2] = cv2.resize(
        bayer * b_mask,
        (w, h),
        interpolation=cv2.INTER_LINEAR
    )
    
    return rgb

# OpenCV 内置方法（更好）
def demosaic_opencv(bayer):
    """使用 OpenCV 的去马赛克"""
    # OpenCV 支持多种 Bayer 模式
    rgb = cv2.cvtColor(bayer, cv2.COLOR_BAYER_RGGB2RGB)
    return rgb

# 对比
bayer = rgb_to_bayer(original_rgb)
rgb_bilinear = demosaic_bilinear(bayer)
rgb_opencv = demosaic_opencv(bayer)

# 评价
psnr_bilinear = calculate_psnr(original_rgb, rgb_bilinear)
psnr_opencv = calculate_psnr(original_rgb, rgb_opencv)
print(f"Bilinear PSNR: {psnr_bilinear:.2f} dB")
print(f"OpenCV PSNR: {psnr_opencv:.2f} dB")
```

#### Day 3: 白平衡（White Balance）

**原理**：

```python
# 白平衡：校正色温，使白色物体在图像中呈现为白色

问题：
不同光源下，相同物体的颜色不同
- 白炽灯：偏黄
- 荧光灯：偏绿
- 日光：中性

解决：
调整 R、G、B 三个通道的增益

RGB_corrected = [R * gain_r, G * gain_g, B * gain_b]

常用算法：
1. Gray World（灰度世界）
   假设：图像平均颜色是灰色
   
2. White Patch（最亮点）
   假设：图像最亮的点是白色
   
3. Learning-based
   使用深度学习预测增益
```

**代码实现**：

```python
def white_balance_gray_world(img):
    """灰度世界白平衡"""
    # 计算各通道平均值
    avg_r = np.mean(img[:,:,0])
    avg_g = np.mean(img[:,:,1])
    avg_b = np.mean(img[:,:,2])
    
    # 计算增益（以 G 通道为参考）
    gain_r = avg_g / avg_r
    gain_b = avg_g / avg_b
    
    # 应用增益
    result = img.copy().astype(np.float32)
    result[:,:,0] *= gain_r
    result[:,:,2] *= gain_b
    
    # 裁剪
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result

def white_balance_white_patch(img):
    """最亮点白平衡"""
    # 找到各通道的最大值
    max_r = np.max(img[:,:,0])
    max_g = np.max(img[:,:,1])
    max_b = np.max(img[:,:,2])
    
    # 计算增益
    gain_r = 255.0 / max_r
    gain_g = 255.0 / max_g
    gain_b = 255.0 / max_b
    
    # 归一化（使最小增益为1）
    min_gain = min(gain_r, gain_g, gain_b)
    gain_r /= min_gain
    gain_g /= min_gain
    gain_b /= min_gain
    
    # 应用
    result = img.copy().astype(np.float32)
    result[:,:,0] *= gain_r
    result[:,:,1] *= gain_g
    result[:,:,2] *= gain_b
    
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result

# 实践
img_tungsten = cv2.imread('tungsten_light.png')  # 白炽灯下拍摄
img_wb_gray = white_balance_gray_world(img_tungsten)
img_wb_white = white_balance_white_patch(img_tungsten)

# 可视化对比
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(cv2.cvtColor(img_tungsten, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original (偏黄)')
axes[1].imshow(cv2.cvtColor(img_wb_gray, cv2.COLOR_BGR2RGB))
axes[1].set_title('Gray World')
axes[2].imshow(cv2.cvtColor(img_wb_white, cv2.COLOR_BGR2RGB))
axes[2].set_title('White Patch')
plt.savefig('white_balance_comparison.png')
```

#### Day 4: Gamma 校正

**原理**：

```python
# Gamma 校正：从线性空间转到非线性空间

为什么需要？
1. 人眼对亮度的感知是非线性的
   - 对暗部变化更敏感
   - 对亮部变化不敏感

2. 显示器也是非线性的

3. 如果直接显示线性数据，图像会很暗

Gamma 校正公式：
Output = Input ^ (1/gamma)

通常 gamma = 2.2 (sRGB 标准)

效果：
- 提亮暗部
- 压缩亮部
- 更符合人眼感知
```

**代码实现**：

```python
def gamma_correction(img, gamma=2.2):
    """Gamma 校正"""
    # 归一化到 [0, 1]
    img_norm = img.astype(np.float32) / 255.0
    
    # 应用 gamma
    img_gamma = np.power(img_norm, 1.0 / gamma)
    
    # 转回 [0, 255]
    result = (img_gamma * 255).clip(0, 255).astype(np.uint8)
    
    return result

def inverse_gamma_correction(img, gamma=2.2):
    """逆 Gamma 校正（用于退化生成）"""
    img_norm = img.astype(np.float32) / 255.0
    img_linear = np.power(img_norm, gamma)
    result = (img_linear * 255).clip(0, 255).astype(np.uint8)
    return result

# 对比
linear_img = cv2.imread('linear.png')  # 假设是线性空间图像
gamma_img = gamma_correction(linear_img, gamma=2.2)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(cv2.cvtColor(linear_img, cv2.COLOR_BGR2RGB))
axes[0].set_title('Linear (很暗)')
axes[1].imshow(cv2.cvtColor(gamma_img, cv2.COLOR_BGR2RGB))
axes[1].set_title('Gamma Corrected (正常)')
plt.savefig('gamma_comparison.png')

# 绘制 Gamma 曲线
x = np.linspace(0, 1, 256)
y_gamma = np.power(x, 1/2.2)

plt.figure(figsize=(8, 6))
plt.plot(x, x, label='Linear (gamma=1.0)', linestyle='--')
plt.plot(x, y_gamma, label='Gamma=2.2')
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Gamma Correction Curve')
plt.legend()
plt.grid(True)
plt.savefig('gamma_curve.png')
```

#### Day 5: 颜色空间转换与色彩校正

**原理**：

```python
# 颜色空间转换

常见颜色空间：
1. RGB - 加法三原色，设备相关
2. sRGB - 标准 RGB，最常用
3. Adobe RGB - 更大色域
4. XYZ - 设备无关的标准空间
5. Lab - 感知均匀的颜色空间

ISP 中的转换链：
Camera RGB → XYZ → sRGB

色彩校正矩阵（CCM）：
RGB_corrected = CCM @ RGB_raw

CCM 通过色卡标定得到
```

**代码实现**：

```python
def color_correction_matrix(img, ccm):
    """
    应用色彩校正矩阵
    
    Args:
        img: numpy array, shape (H, W, 3)
        ccm: numpy array, shape (3, 3), 色彩校正矩阵
    """
    h, w, c = img.shape
    
    # 转为浮点数
    img_float = img.astype(np.float32) / 255.0
    
    # 重塑为 (N, 3)
    img_flat = img_float.reshape(-1, 3)
    
    # 应用矩阵
    img_corrected = img_flat @ ccm.T
    
    # 重塑回原形状
    img_corrected = img_corrected.reshape(h, w, c)
    
    # 裁剪并转回 uint8
    img_corrected = np.clip(img_corrected, 0, 1)
    result = (img_corrected * 255).astype(np.uint8)
    
    return result

# 示例 CCM（简化版）
ccm_example = np.array([
    [1.2, -0.1, -0.1],
    [-0.1, 1.1, 0.0],
    [0.0, -0.2, 1.2]
])

img = cv2.imread('raw_rgb.png')
img_corrected = color_correction_matrix(img, ccm_example)

# 颜色空间转换
def rgb_to_xyz(rgb):
    """RGB 转 XYZ（sRGB 标准）"""
    # 转换矩阵
    M = np.array([
        [0.4124, 0.3576, 0.1805],
        [0.2126, 0.7152, 0.0722],
        [0.0193, 0.1192, 0.9505]
    ])
    
    rgb_norm = rgb.astype(np.float32) / 255.0
    h, w, c = rgb_norm.shape
    rgb_flat = rgb_norm.reshape(-1, 3)
    xyz_flat = rgb_flat @ M.T
    xyz = xyz_flat.reshape(h, w, c)
    
    return xyz

def xyz_to_lab(xyz):
    """XYZ 转 Lab"""
    # 标准光源 D65
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    
    x = xyz[:,:,0] / Xn
    y = xyz[:,:,1] / Yn
    z = xyz[:,:,2] / Zn
    
    def f(t):
        delta = 6/29
        return np.where(
            t > delta**3,
            np.power(t, 1/3),
            t / (3 * delta**2) + 4/29
        )
    
    fx = f(x)
    fy = f(y)
    fz = f(z)
    
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    
    return np.stack([L, a, b], axis=-1)

# 使用
rgb = cv2.imread('test.png')
rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
xyz = rgb_to_xyz(rgb)
lab = xyz_to_lab(xyz)
```

#### Day 6-7: 完整 ISP 流程实现

**综合实践**：

```python
"""
完整的 ISP 流程实现
"""

class SimplifiedISP:
    """简化的 ISP 流程"""
    def __init__(self, ccm=None, gamma=2.2):
        self.ccm = ccm if ccm is not None else np.eye(3)
        self.gamma = gamma
    
    def process(self, bayer_img):
        """
        完整处理流程
        
        Input: Bayer 图像 (H, W)
        Output: RGB 图像 (H, W, 3)
        """
        print("ISP 处理流程:")
        
        # 1. 去马赛克
        print("  1. 去马赛克...")
        rgb = cv2.cvtColor(bayer_img, cv2.COLOR_BAYER_RGGB2RGB)
        
        # 2. 去噪（简化，实际更复杂）
        print("  2. 去噪...")
        rgb = cv2.fastNlMeansDenoisingColored(rgb, None, 10, 10, 7, 21)
        
        # 3. 白平衡
        print("  3. 白平衡...")
        rgb = self.white_balance(rgb)
        
        # 4. 色彩校正
        print("  4. 色彩校正...")
        rgb = self.color_correction(rgb)
        
        # 5. Gamma 校正
        print("  5. Gamma 校正...")
        rgb = self.gamma_correction(rgb)
        
        # 6. 锐化（可选）
        print("  6. 锐化...")
        rgb = self.sharpen(rgb)
        
        print("ISP 处理完成")
        return rgb
    
    def white_balance(self, img):
        """白平衡"""
        avg_r = np.mean(img[:,:,0])
        avg_g = np.mean(img[:,:,1])
        avg_b = np.mean(img[:,:,2])
        
        gain_r = avg_g / avg_r
        gain_b = avg_g / avg_b
        
        result = img.copy().astype(np.float32)
        result[:,:,0] *= gain_r
        result[:,:,2] *= gain_b
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    def color_correction(self, img):
        """色彩校正"""
        h, w, c = img.shape
        img_float = img.astype(np.float32) / 255.0
        img_flat = img_float.reshape(-1, 3)
        img_corrected = img_flat @ self.ccm.T
        img_corrected = img_corrected.reshape(h, w, c)
        img_corrected = np.clip(img_corrected, 0, 1)
        return (img_corrected * 255).astype(np.uint8)
    
    def gamma_correction(self, img):
        """Gamma 校正"""
        img_norm = img.astype(np.float32) / 255.0
        img_gamma = np.power(img_norm, 1.0 / self.gamma)
        return (img_gamma * 255).clip(0, 255).astype(np.uint8)
    
    def sharpen(self, img):
        """锐化"""
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ])
        return cv2.filter2D(img, -1, kernel)

# 测试完整流程
rgb_original = cv2.imread('original.png')
bayer = rgb_to_bayer(rgb_original)  # 模拟 Bayer 图像

isp = SimplifiedISP()
rgb_processed = isp.process(bayer)

# 对比
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(bayer, cmap='gray')
axes[0].set_title('Input: Bayer')
axes[1].imshow(cv2.cvtColor(rgb_processed, cv2.COLOR_BGR2RGB))
axes[1].set_title('Output: ISP Processed')
axes[2].imshow(cv2.cvtColor(rgb_original, cv2.COLOR_BGR2RGB))
axes[2].set_title('Reference: Original')
plt.savefig('isp_complete_pipeline.png')

# 评价
psnr = calculate_psnr(rgb_original, rgb_processed)
ssim_score = calculate_ssim_color(rgb_original, rgb_processed)
print(f"\nPSNR: {psnr:.2f} dB")
print(f"SSIM: {ssim_score:.4f}")
```

### 2.3 阶段总结与检验

**自我检验清单**：

```
□ 理解 Bayer 阵列的原理和作用
□ 能实现简单的去马赛克算法
□ 理解白平衡的必要性和常用方法
□ 理解 Gamma 校正的作用
□ 能搭建完整的 ISP 流程
□ 理解 ISP 各模块之间的依赖关系
```

**实践项目**：

```python
# 项目：逆 ISP 用于数据退化
class InverseISP:
    """逆 ISP：从 RGB 生成类 Raw 图像"""
    def __init__(self, gamma=2.2):
        self.gamma = gamma
    
    def process(self, rgb_img):
        """逆 ISP 流程"""
        # 1. 逆 Gamma
        linear = self.inverse_gamma(rgb_img)
        
        # 2. 添加噪声（模拟传感器噪声）
        noisy = self.add_sensor_noise(linear)
        
        # 3. 逆颜色校正（可选）
        # ...
        
        # 4. 逆白平衡（可选）
        # ...
        
        # 5. 转为 Bayer（可选，用于某些任务）
        # bayer = self.rgb_to_bayer(noisy)
        
        return noisy
    
    def inverse_gamma(self, img):
        """逆 Gamma"""
        img_norm = img.astype(np.float32) / 255.0
        img_linear = np.power(img_norm, self.gamma)
        return (img_linear * 255).astype(np.uint8)
    
    def add_sensor_noise(self, img):
        """添加传感器噪声"""
        # 泊松噪声（信号相关）
        img_float = img.astype(np.float32)
        noisy = np.random.poisson(img_float * 0.01) / 0.01
        
        # 高斯噪声（读取噪声）
        noisy += np.random.normal(0, 5, img.shape)
        
        return np.clip(noisy, 0, 255).astype(np.uint8)

# 使用逆 ISP 生成训练数据
inverse_isp = InverseISP()

clean_rgb = cv2.imread('clean.png')
degraded = inverse_isp.process(clean_rgb)

# 训练 AI ISP 复原网络
# model(degraded) → clean_rgb
```

---

## 3. 第三阶段：图像退化模型（Week 3-4）

> **目标**：深入理解各种退化类型，能设计真实的退化模型  
> **时间**：10-14天  
> **难度**：⭐⭐⭐⭐

### 3.1 学习路线

```
Week 3:
├─ Day 1-2: 噪声模型深入
├─ Day 3-4: 模糊模型深入
├─ Day 5-6: 组合退化与真实退化建模
└─ Day 7: Real-ESRGAN 论文精读

Week 4:
├─ Day 1-3: 复现退化模型代码
├─ Day 4-5: 构建自己的退化数据集
└─ Day 6-7: 验证退化模型的真实性
```

### 3.2 核心内容

#### Week 3, Day 1-2: 噪声模型深入

**必读论文**：
- "A Comprehensive Analysis of Random Noise Models and the Gaussian Assumption"

**关键知识点**：

```python
# 1. 传感器噪声模型
总噪声 = 光子噪声 + 暗电流噪声 + 读取噪声

I_noisy = Poisson(I_clean) + Gaussian(0, σ_read^2)

# 2. ISO 感光度与噪声的关系
噪声强度 ∝ √ISO

# 3. 不同光照下的噪声
低光照: 噪声占比高（SNR 低）
高光照: 噪声占比低（SNR 高）

# 4. 真实相机噪声标定
使用多张同场景照片估计噪声参数
```

**实践代码**：

```python
class CameraNoise Model:
    """真实相机噪声模型"""
    def __init__(self, iso=400, read_noise=1.5):
        self.iso = iso
        self.read_noise = read_noise
        self.base_iso = 100
    
    def add_noise(self, img):
        """添加真实相机噪声"""
        # 归一化
        img_norm = img.astype(np.float32) / 255.0
        
        # 光子噪声（泊松）- 与信号强度相关
        # ISO 越高，增益越大，噪声越明显
        gain = self.iso / self.base_iso
        img_photon = np.random.poisson(img_norm * 255 * gain) / gain / 255
        
        # 读取噪声（高斯）- 与信号无关
        read_noise_scaled = self.read_noise * gain / 255
        noise_read = np.random.normal(0, read_noise_scaled, img.shape)
        
        # 合成
        img_noisy = img_photon + noise_read
        
        # 裁剪
        img_noisy = np.clip(img_noisy, 0, 1)
        
        return (img_noisy * 255).astype(np.uint8)

# 测试不同 ISO
clean = cv2.imread('clean.png')
noise_model = CameraNoisModel()

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for i, iso in enumerate([100, 400, 1600, 3200, 6400, 12800]):
    noise_model.iso = iso
    noisy = noise_model.add_noise(clean)
    
    ax = axes[i//3, i%3]
    ax.imshow(cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB))
    ax.set_title(f'ISO {iso}')
    ax.axis('off')

plt.tight_layout()
plt.savefig('noise_iso_comparison.png')
```

#### Week 3, Day 3-4: 模糊模型深入

**必读资源**：
- "Understanding and Evaluating Blind Deconvolution Algorithms"
- "Deep Learning for Understanding Image Blur"

**关键知识点**：

```python
# 模糊核（Point Spread Function, PSF）

1. 运动模糊
   - 相机抖动
   - 物体运动
   - PSF: 直线型

2. 散焦模糊
   - 对焦不准
   - 景深效果
   - PSF: 圆盘型（disk blur）

3. 大气湍流模糊
   - 空气扰动
   - PSF: 各向同性高斯

4. 组合模糊
   - 多种模糊叠加
```

**实践代码**：

```python
def generate_motion_blur_kernel(length, angle):
    """生成运动模糊核"""
    # 创建空白核
    kernel = np.zeros((length, length))
    
    # 中心线
    center = length // 2
    kernel[center, :] = 1
    
    # 旋转
    M = cv2.getRotationMatrix2D((center, center), angle, 1.0)
    kernel = cv2.warpAffine(kernel, M, (length, length))
    
    # 归一化
    kernel = kernel / kernel.sum()
    
    return kernel

def generate_defocus_blur_kernel(radius):
    """生成散焦模糊核（圆盘）"""
    size = 2 * radius + 1
    kernel = np.zeros((size, size))
    
    # 创建圆盘
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x**2 + y**2 <= radius**2
    kernel[mask] = 1
    
    # 归一化
    kernel = kernel / kernel.sum()
    
    return kernel

def apply_blur(img, kernel):
    """应用模糊核"""
    return cv2.filter2D(img, -1, kernel)

# 测试各种模糊
clean = cv2.imread('clean.png')

# 运动模糊（不同角度和长度）
motion_kernels = [
    (15, 0),    # 水平
    (15, 45),   # 对角
    (15, 90),   # 垂直
    (25, 30),   # 更长，倾斜
]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for i, (length, angle) in enumerate(motion_kernels):
    kernel = generate_motion_blur_kernel(length, angle)
    blurred = apply_blur(clean, kernel)
    
    ax = axes[i//3, i%3]
    ax.imshow(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
    ax.set_title(f'Motion: L={length}, θ={angle}°')
    ax.axis('off')

# 散焦模糊
for i, radius in enumerate([3, 5, 7]):
    kernel = generate_defocus_blur_kernel(radius)
    blurred = apply_blur(clean, kernel)
    
    ax = axes[1, i]
    ax.imshow(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
    ax.set_title(f'Defocus: r={radius}')
    ax.axis('off')

plt.tight_layout()
plt.savefig('blur_types.png')
```

#### Week 3, Day 5-7: 组合退化与 Real-ESRGAN

**必读论文**：
- **Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data** (ICCV 2021)

**核心思想**：

```python
# Real-ESRGAN 的退化模型（二阶退化）

第一阶退化流程：
clean_img
  → [Blur1] → [Downsample] → [Noise1] → [JPEG]
  → intermediate_img

第二阶退化流程：
intermediate_img
  → [Blur2] → [Downsample] → [Noise2] → [JPEG]
  → final_degraded_img

关键创新：
✅ 二阶退化更真实
✅ sinc 滤波器（模拟下采样伪影）
✅ 随机顺序（有时先模糊，有时先噪声）
✅ JPEG 压缩两次（更真实）

为什么这样做？
现实中的退化图像通常经历多次处理：
拍摄 → 压缩 → 传输 → 再压缩 → ...
```

**代码实现（简化版 Real-ESRGAN 退化）**：

```python
class RealESRGANDegradation:
    """Real-ESRGAN 风格的退化模型"""
    def __init__(self, scale=4):
        self.scale = scale
    
    def degrade(self, img):
        """
        二阶退化流程
        
        Args:
            img: numpy array, 高分辨率图像
        
        Returns:
            degraded: 低分辨率退化图像
        """
        # ========== 第一阶退化 ==========
        # 1. 模糊
        blur_type1 = np.random.choice(['iso', 'aniso'])
        if blur_type1 == 'iso':
            kernel1 = self.generate_isotropic_kernel()
        else:
            kernel1 = self.generate_anisotropic_kernel()
        img = cv2.filter2D(img, -1, kernel1)
        
        # 2. 下采样
        h, w = img.shape[:2]
        img = cv2.resize(img, (w//2, h//2), interpolation=cv2.INTER_AREA)
        
        # 3. 噪声
        noise_level1 = np.random.uniform(1, 15)
        img = self.add_gaussian_noise(img, noise_level1)
        
        # 4. JPEG 压缩
        quality1 = np.random.randint(60, 95)
        img = self.jpeg_compress(img, quality1)
        
        # ========== 第二阶退化 ==========
        # 5. 再次模糊
        blur_type2 = np.random.choice(['iso', 'aniso'])
        if blur_type2 == 'iso':
            kernel2 = self.generate_isotropic_kernel()
        else:
            kernel2 = self.generate_anisotropic_kernel()
        img = cv2.filter2D(img, -1, kernel2)
        
        # 6. 再次下采样
        h, w = img.shape[:2]
        img = cv2.resize(img, (w//2, h//2), interpolation=cv2.INTER_AREA)
        
        # 7. 再次噪声
        noise_level2 = np.random.uniform(1, 15)
        img = self.add_gaussian_noise(img, noise_level2)
        
        # 8. 再次 JPEG
        quality2 = np.random.randint(40, 80)
        img = self.jpeg_compress(img, quality2)
        
        # 9. sinc 滤波器（模拟下采样伪影）
        if np.random.rand() > 0.5:
            img = self.apply_sinc_filter(img)
        
        return img
    
    def generate_isotropic_kernel(self):
        """生成各向同性高斯核"""
        kernel_size = np.random.choice([7, 9, 11, 13, 15])
        sigma = np.random.uniform(0.1, 2.4)
        
        kernel = cv2.getGaussianKernel(kernel_size, sigma)
        kernel = kernel @ kernel.T
        
        return kernel
    
    def generate_anisotropic_kernel(self):
        """生成各向异性高斯核"""
        kernel_size = np.random.choice([7, 9, 11, 13, 15])
        sigma_x = np.random.uniform(0.5, 6)
        sigma_y = np.random.uniform(0.5, 6)
        theta = np.random.uniform(0, np.pi)
        
        # ... (复杂的各向异性核生成代码)
        # 简化版：直接返回各向同性
        return self.generate_isotropic_kernel()
    
    def add_gaussian_noise(self, img, sigma):
        """添加高斯噪声"""
        noise = np.random.normal(0, sigma, img.shape)
        noisy = img + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    def jpeg_compress(self, img, quality):
        """JPEG 压缩"""
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode('.jpg', img, encode_param)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        return decoded
    
    def apply_sinc_filter(self, img):
        """sinc 滤波器（简化版）"""
        # 实际实现较复杂，这里用简单的锐化代替
        kernel = np.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ]) / 9
        return cv2.filter2D(img, -1, kernel)

# 使用
degradation_model = RealESRGANDegradation(scale=4)

hr_img = cv2.imread('high_res.png')  # 1024x1024
lr_degraded = degradation_model.degrade(hr_img)  # ~256x256

print(f"HR shape: {hr_img.shape}")
print(f"LR shape: {lr_degraded.shape}")

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB))
axes[0].set_title('High-Resolution')
axes[1].imshow(cv2.cvtColor(lr_degraded, cv2.COLOR_BGR2RGB))
axes[1].set_title('Degraded (Real-ESRGAN style)')
plt.savefig('real_esrgan_degradation.png')
```

### 3.3 Week 4: 实践与验证

**核心任务**：

```python
# 任务1：构建退化数据集

class DegradationDataset(Dataset):
    """退化数据集（综合所有退化类型）"""
    def __init__(self, clean_dir, degradation_types='all'):
        self.clean_paths = glob(os.path.join(clean_dir, '*.png'))
        self.degradation_types = degradation_types
        
        # 退化模型
        self.real_esrgan_deg = RealESRGANDegradation()
        self.camera_noise = CameraNoiseModel()
    
    def __getitem__(self, idx):
        # 读取干净图像
        clean = cv2.imread(self.clean_paths[idx])
        clean = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)
        
        # 随机选择退化类型
        deg_type = np.random.choice([
            'real_esrgan',
            'camera_noise',
            'simple_blur_noise',
            'jpeg_only'
        ])
        
        if deg_type == 'real_esrgan':
            degraded = self.real_esrgan_deg.degrade(clean)
        elif deg_type == 'camera_noise':
            degraded = self.camera_noise.add_noise(clean)
        # ... 其他退化类型
        
        # 转为 Tensor
        clean_tensor = self.to_tensor(clean)
        degraded_tensor = self.to_tensor(degraded)
        
        return degraded_tensor, clean_tensor
    
    def __len__(self):
        return len(self.clean_paths)

# 任务2：验证退化模型的真实性

def evaluate_degradation_realism(synthetic_degraded, real_degraded):
    """
    评估合成退化与真实退化的相似性
    
    方法：
    1. 统计特性对比（噪声水平、频谱）
    2. 感知相似性（LPIPS）
    3. 在真实数据上测试模型泛化性
    """
    # 1. 噪声水平估计
    syn_noise = estimate_noise_level(synthetic_degraded)
    real_noise = estimate_noise_level(real_degraded)
    
    print(f"合成退化噪声水平: {syn_noise:.2f}")
    print(f"真实退化噪声水平: {real_noise:.2f}")
    
    # 2. 频谱分析
    syn_spectrum = np.fft.fft2(cv2.cvtColor(synthetic_degraded, cv2.COLOR_BGR2GRAY))
    real_spectrum = np.fft.fft2(cv2.cvtColor(real_degraded, cv2.COLOR_BGR2GRAY))
    
    # ... 对比频谱特性
    
    # 3. 在真实数据上测试
    # 用合成数据训练的模型在真实数据上的表现

def estimate_noise_level(img):
    """估计图像噪声水平"""
    # 简化方法：高频成分的标准差
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    noise_sigma = np.std(laplacian)
    return noise_sigma
```

### 3.4 阶段总结

**知识点检验**：

```
□ 理解真实相机噪声的组成
□ 能实现复杂的模糊核生成
□ 理解 Real-ESRGAN 的二阶退化思想
□ 能设计自己的退化模型
□ 能验证退化模型的真实性
```

---

## 4. 第四阶段：感知损失与高级技术（Week 5）

> **目标**：理解感知损失、对抗损失等高级技术  
> **时间**：5-7天  
> **难度**：⭐⭐⭐

### 4.1 为什么需要感知损失？

```python
# 传统损失函数的问题

L1/L2 Loss:
  优点：✅ 简单，易优化，PSNR 高
  缺点：❌ 过度平滑，缺少高频细节，感知质量差

问题案例：
两张图像，像素平均误差相同，但：
- 图A：略微模糊，但结构完整
- 图B：锐利，但有细微纹理差异

L1/L2 认为两者相同，但人眼更喜欢图B

解决方案：感知损失（Perceptual Loss）
使用预训练网络的特征表示来衡量相似性
```

### 4.2 学习内容

#### Day 1-2: 感知损失原理与实现

**必读论文**：
- **"Perceptual Losses for Real-Time Style Transfer and Super-Resolution"** (ECCV 2016)
- "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric" (LPIPS, CVPR 2018)

**核心思想**：

```python
# 感知损失定义

L_perceptual = ||φ(I_pred) - φ(I_gt)||^2

其中：
- φ: 预训练网络（如 VGG）的特征提取器
- I_pred: 网络预测的图像
- I_gt: Ground Truth

为什么有效？
✅ 深度特征捕捉了高层语义信息
✅ 更符合人类视觉感知
✅ 鼓励生成细节和纹理
```

**代码实现**：

```python
import torch
import torch.nn as nn
import torchvision.models as models

class VGGPerceptualLoss(nn.Module):
    """VGG 感知损失"""
    def __init__(self, layers=['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']):
        super().__init__()
        
        # 加载预训练 VGG16
        vgg = models.vgg16(pretrained=True).features
        
        # 提取特定层
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        
        # relu1_2: layer 4
        for x in range(4):
            self.slice1.add_module(str(x), vgg[x])
        
        # relu2_2: layer 9
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg[x])
        
        # relu3_3: layer 16
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg[x])
        
        # relu4_3: layer 23
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg[x])
        
        # 冻结参数
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, pred, target):
        """
        计算感知损失
        
        Args:
            pred: 预测图像，shape (B, 3, H, W)
            target: 目标图像，shape (B, 3, H, W)
        """
        # 归一化（VGG 预训练时的归一化）
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred.device)
        
        pred = (pred - mean) / std
        target = (target - mean) / std
        
        # 提取特征
        pred_relu1_2 = self.slice1(pred)
        pred_relu2_2 = self.slice2(pred_relu1_2)
        pred_relu3_3 = self.slice3(pred_relu2_2)
        pred_relu4_3 = self.slice4(pred_relu3_3)
        
        target_relu1_2 = self.slice1(target)
        target_relu2_2 = self.slice2(target_relu1_2)
        target_relu3_3 = self.slice3(target_relu2_2)
        target_relu4_3 = self.slice4(target_relu3_3)
        
        # 计算各层损失
        loss1 = nn.functional.l1_loss(pred_relu1_2, target_relu1_2)
        loss2 = nn.functional.l1_loss(pred_relu2_2, target_relu2_2)
        loss3 = nn.functional.l1_loss(pred_relu3_3, target_relu3_3)
        loss4 = nn.functional.l1_loss(pred_relu4_3, target_relu4_3)
        
        # 加权求和
        loss = loss1 + loss2 + loss3 + loss4
        
        return loss

# 使用示例
perceptual_loss_fn = VGGPerceptualLoss()

# 训练中
for degraded, clean in train_loader:
    restored = model(degraded)
    
    # 组合损失
    l1_loss = nn.functional.l1_loss(restored, clean)
    perceptual_loss = perceptual_loss_fn(restored, clean)
    
    total_loss = l1_loss + 0.1 * perceptual_loss  # 0.1 是权重
    
    total_loss.backward()
    optimizer.step()
```

#### Day 3-4: 对抗损失（GAN Loss）

**必读论文**：
- "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network" (SRGAN, CVPR 2017)
- "ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks" (ECCV 2018)

**核心思想**：

```python
# GAN 用于图像复原

生成器 G:
  输入：退化图像
  输出：复原图像

判别器 D:
  输入：图像（真实或生成）
  输出：真/假概率

训练目标：
  G: 欺骗 D（让 D 认为生成图像是真实的）
  D: 区分真假（识别生成图像）

损失函数：
  L_G = L_pixel + λ_perceptual * L_perceptual + λ_adv * L_adversarial
  L_D = -log(D(real)) - log(1 - D(fake))

优势：
✅ 生成更真实的纹理和细节
✅ 提升感知质量

挑战：
❌ 训练不稳定
❌ 可能产生伪影
❌ PSNR 可能下降
```

**代码实现**：

```python
import torch
import torch.nn as nn

class Discriminator(nn.Module):
    """判别器（简化版）"""
    def __init__(self):
        super().__init__()
        
        self.net = nn.Sequential(
            # 输入：3 x 128 x 128
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 1024, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 1, 1)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        return self.net(x).view(batch_size)

# GAN 训练循环
generator = YourGenerator()
discriminator = Discriminator()

optimizer_G = torch.optim.Adam(generator.parameters(), lr=1e-4)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

l1_loss = nn.L1Loss()
perceptual_loss_fn = VGGPerceptualLoss()
adversarial_loss = nn.BCEWithLogitsLoss()

for epoch in range(num_epochs):
    for degraded, clean in train_loader:
        degraded = degraded.cuda()
        clean = clean.cuda()
        batch_size = clean.size(0)
        
        # ========== 训练判别器 ==========
        optimizer_D.zero_grad()
        
        # 真实图像
        real_labels = torch.ones(batch_size).cuda()
        real_output = discriminator(clean)
        d_loss_real = adversarial_loss(real_output, real_labels)
        
        # 生成图像
        fake_labels = torch.zeros(batch_size).cuda()
        with torch.no_grad():
            fake_images = generator(degraded)
        fake_output = discriminator(fake_images.detach())
        d_loss_fake = adversarial_loss(fake_output, fake_labels)
        
        # 判别器总损失
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()
        
        # ========== 训练生成器 ==========
        optimizer_G.zero_grad()
        
        # 生成图像
        fake_images = generator(degraded)
        
        # 像素损失
        pixel_loss = l1_loss(fake_images, clean)
        
        # 感知损失
        percept_loss = perceptual_loss_fn(fake_images, clean)
        
        # 对抗损失（欺骗判别器）
        fake_output = discriminator(fake_images)
        adv_loss = adversarial_loss(fake_output, real_labels)
        
        # 生成器总损失
        g_loss = pixel_loss + 0.1 * percept_loss + 0.001 * adv_loss
        
        g_loss.backward()
        optimizer_G.step()
        
        # 打印
        if i % 100 == 0:
            print(f"Epoch [{epoch}], D Loss: {d_loss.item():.4f}, "
                  f"G Loss: {g_loss.item():.4f}")
```

#### Day 5: 其他高级技术

**1. Charbonnier Loss（更平滑的 L1）**

```python
class CharbonnierLoss(nn.Module):
    """Charbonnier 损失（NAFNet 等使用）"""
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    
    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps)
        return torch.mean(loss)

# 使用
loss_fn = CharbonnierLoss()
loss = loss_fn(restored, clean)
```

**2. Frequency Loss（频域损失）**

```python
class FrequencyLoss(nn.Module):
    """频域损失"""
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        # FFT
        pred_fft = torch.fft.fft2(pred)
        target_fft = torch.fft.fft2(target)
        
        # 频谱幅度
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        
        # L1 损失
        loss = torch.mean(torch.abs(pred_mag - target_mag))
        
        return loss

# 使用（通常与像素损失组合）
pixel_loss = l1_loss(restored, clean)
freq_loss = frequency_loss(restored, clean)
total_loss = pixel_loss + 0.1 * freq_loss
```

### 4.3 阶段总结

**损失函数选择指南**：

```python
任务类型 → 推荐损失函数

去噪：
  L1 或 Charbonnier（主要）
  + Perceptual Loss（可选，提升感知质量）

超分辨率：
  L1（PSNR 导向）
  L1 + Perceptual + GAN（感知质量导向）

去模糊：
  L1 + Perceptual
  + Frequency Loss（恢复高频细节）

AI ISP：
  L1 + Perceptual
  可选 GAN（追求极致感知质量）

通用建议：
  1. 先用简单的 L1/L2 验证模型能力
  2. 加入 Perceptual Loss 提升感知质量
  3. 谨慎使用 GAN（训练难度大）
```

---

## 5. 第五阶段：实战项目（Week 6-8）

> **目标**：完整实现一个 AI ISP 或图像复原项目  
> **时间**：15-21天  
> **难度**：⭐⭐⭐⭐⭐

### 5.1 项目选择

**推荐项目（选其一）**：

```
项目1：图像去噪网络（入门）
├─ 数据：DIV2K + 合成噪声
├─ 网络：NAFNet 或 Restormer
├─ 指标：PSNR > 30 dB
└─ 时间：2周

项目2：图像超分辨率（进阶）
├─ 数据：DIV2K
├─ 网络：SwinIR 或 HAT
├─ 指标：PSNR > 32 dB (2x)
└─ 时间：2-3周

项目3：AI ISP 去噪增强（高级）
├─ 数据：自己拍摄 + 合成退化
├─ 网络：NAFNet + 自定义
├─ 指标：PSNR + 视觉质量
└─ 时间：3周

项目4：真实退化复原（挑战）
├─ 数据：Real-ESRGAN 数据集
├─ 网络：Restormer + GAN
├─ 指标：感知质量优先
└─ 时间：3-4周
```

### 5.2 项目实施计划（以项目3为例）

#### Week 6: 数据准备

```python
# Day 1-2: 收集高质量图像
# - 用相机拍摄（各种场景、光照）
# - 或下载数据集（Flickr2K, DIV2K）

# Day 3-4: 实现退化管道
class AI_ISP_DegradationPipeline:
    """AI ISP 退化管道"""
    def __init__(self):
        self.camera_noise = CameraNoiseModel()
        # ... 其他退化模型
    
    def degrade(self, clean_img):
        """
        模拟 AI ISP 的输入数据
        
        退化流程：
        1. 逆 Gamma
        2. 添加传感器噪声
        3. 模拟低照度（可选）
        4. 添加轻微模糊（可选）
        """
        # 实现...
        pass

# Day 5-7: 构建数据集并验证
dataset = AI_ISP_Dataset(clean_dir='./data/clean')
loader = DataLoader(dataset, batch_size=16)

# 验证退化效果
for degraded, clean in loader:
    visualize_batch(degraded, clean)
    break
```

#### Week 7: 模型训练

```python
# Day 1-3: 搭建训练框架

class Trainer:
    """训练器"""
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # 损失函数
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = VGGPerceptualLoss()
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=2e-4,
            betas=(0.9, 0.99)
        )
        
        # 学习率调度
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=200
        )
    
    def train_epoch(self):
        """训练一个 epoch"""
        self.model.train()
        epoch_loss = 0
        
        for degraded, clean in self.train_loader:
            degraded = degraded.cuda()
            clean = clean.cuda()
            
            # 前向
            restored = self.model(degraded)
            
            # 损失
            l1 = self.l1_loss(restored, clean)
            perceptual = self.perceptual_loss(restored, clean)
            loss = l1 + 0.1 * perceptual
            
            # 反向
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
        
        return epoch_loss / len(self.train_loader)
    
    def validate(self):
        """验证"""
        self.model.eval()
        total_psnr = 0
        total_ssim = 0
        
        with torch.no_grad():
            for degraded, clean in self.val_loader:
                degraded = degraded.cuda()
                clean = clean.cuda()
                
                restored = self.model(degraded)
                
                # 计算指标
                psnr = calculate_psnr(restored, clean)
                ssim = calculate_ssim(restored, clean)
                
                total_psnr += psnr
                total_ssim += ssim
        
        avg_psnr = total_psnr / len(self.val_loader)
        avg_ssim = total_ssim / len(self.val_loader)
        
        return avg_psnr, avg_ssim
    
    def train(self, num_epochs):
        """完整训练"""
        best_psnr = 0
        
        for epoch in range(num_epochs):
            # 训练
            train_loss = self.train_epoch()
            
            # 验证
            val_psnr, val_ssim = self.validate()
            
            # 学习率调整
            self.scheduler.step()
            
            # 打印
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val PSNR: {val_psnr:.2f} dB")
            print(f"  Val SSIM: {val_ssim:.4f}")
            
            # 保存最佳模型
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save(
                    self.model.state_dict(),
                    f'best_model_psnr{val_psnr:.2f}.pth'
                )
                print(f"  ✅ 保存最佳模型")

# Day 4-7: 训练模型
model = NAFNet()
trainer = Trainer(model, train_loader, val_loader)
trainer.train(num_epochs=200)
```

#### Week 8: 测试与优化

```python
# Day 1-2: 测试和可视化

def test_and_visualize(model_path, test_dir):
    """测试并可视化结果"""
    # 加载模型
    model = NAFNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.cuda()
    
    # 测试图像
    test_images = glob(os.path.join(test_dir, '*.png'))
    
    results = []
    
    for img_path in test_images:
        # 读取
        clean = cv2.imread(img_path)
        
        # 生成退化
        degraded = degradation_model.degrade(clean)
        
        # 复原
        degraded_tensor = to_tensor(degraded).unsqueeze(0).cuda()
        with torch.no_grad():
            restored_tensor = model(degraded_tensor)
        restored = tensor_to_image(restored_tensor)
        
        # 评价
        psnr_degraded = calculate_psnr(clean, degraded)
        psnr_restored = calculate_psnr(clean, restored)
        
        ssim_degraded = calculate_ssim(clean, degraded)
        ssim_restored = calculate_ssim(clean, restored)
        
        # 可视化
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(degraded)
        axes[0].set_title(f'Degraded\nPSNR: {psnr_degraded:.2f}')
        
        axes[1].imshow(restored)
        axes[1].set_title(f'Restored\nPSNR: {psnr_restored:.2f}')
        
        axes[2].imshow(clean)
        axes[2].set_title('Clean')
        
        plt.savefig(f'result_{os.path.basename(img_path)}')
        
        results.append({
            'image': os.path.basename(img_path),
            'psnr_degraded': psnr_degraded,
            'psnr_restored': psnr_restored,
            'psnr_gain': psnr_restored - psnr_degraded
        })
    
    # 统计
    import pandas as pd
    df = pd.DataFrame(results)
    print(df)
    print(f"\n平均 PSNR 提升: {df['psnr_gain'].mean():.2f} dB")

# Day 3-5: 模型量化（结合之前的量化知识）

from aimet_torch.quantsim import QuantizationSimModel

# 创建量化模拟器
quant_sim = QuantizationSimModel(
    model=model,
    dummy_input=torch.randn(1, 3, 256, 256).cuda(),
    default_param_bw=8,
    default_output_bw=8
)

# 校准
quant_sim.compute_encodings(calibrate, 200)

# 评估量化后精度
quant_psnr = evaluate_psnr(quant_sim.model, test_loader)
print(f"量化后 PSNR: {quant_psnr:.2f} dB")

# 如果精度下降明显，使用 QAT
# ...

# Day 6-7: 整理文档和代码
```

### 5.3 项目交付物

```
项目交付清单：

□ 代码
  ├─ 退化模型实现
  ├─ 数据集构建代码
  ├─ 训练代码
  ├─ 测试和评估代码
  └─ 量化部署代码

□ 模型
  ├─ FP32 模型
  ├─ INT8 量化模型
  └─ 训练日志和曲线

□ 结果
  ├─ 测试集评价指标
  ├─ 可视化对比图
  └─ 在真实数据上的测试

□ 文档
  ├─ 项目总结
  ├─ 技术难点和解决方案
  └─ 后续优化方向
```

---

## 6. 进阶路线：成为专家

完成上述 5 个阶段后，您已经具备了 AI ISP 的基础能力。继续进阶：

### 6.1 深入方向

```
方向1：Transformer 在图像复原中的应用
├─ Restormer
├─ SwinIR
├─ HAT (Hybrid Attention Transformer)
└─ 时间：2-3周

方向2：视频复原
├─ BasicVSR
├─ RVRT (Recurrent Video Restoration Transformer)
└─ 时间：3-4周

方向3：真实场景 AI ISP
├─ Raw 到 RGB 端到端学习
├─ 联合去噪、去马赛克、白平衡
└─ 时间：4-6周

方向4：高效模型设计
├─ 轻量级网络（MobileNet 风格）
├─ 知识蒸馏
├─ 神经架构搜索（NAS）
└─ 时间：4-8周
```

### 6.2 持续学习

```
每周：
├─ 阅读 1-2 篇最新论文（arXiv, CVPR, ICCV）
├─ 复现一个新方法的核心代码
└─ 在自己的数据上测试

每月：
├─ 完成一个小项目或实验
├─ 总结技术博客
└─ 参与开源社区（GitHub）

每季度：
├─ 尝试新的研究方向
├─ 投稿会议/竞赛
└─ 与同行交流分享
```

---

## 7. 资源汇总

### 7.1 论文（按重要性排序）

**必读（★★★★★）**：
1. Real-ESRGAN (ICCV 2021)
2. NAFNet (ECCV 2022)
3. Perceptual Losses (ECCV 2016)
4. LPIPS (CVPR 2018)

**推荐（★★★★）**：
5. Restormer (CVPR 2022)
6. SwinIR (ICCV 2021)
7. SRGAN/ESRGAN
8. DnCNN (TIP 2017)

**进阶（★★★）**：
9. MPRNet (CVPR 2021)
10. HINet (CVPR 2021)

### 7.2 代码仓库

```
1. BasicSR (★★★★★)
   https://github.com/XPixelGroup/BasicSR
   → 最全面的图像复原工具箱

2. NAFNet (★★★★★)
   https://github.com/megvii-research/NAFNet
   → 简洁高效，适合入门

3. Real-ESRGAN (★★★★)
   https://github.com/xinntao/Real-ESRGAN
   → 真实退化建模的优秀实现

4. Restormer (★★★★)
   https://github.com/swz30/Restormer
   → Transformer 复原网络

5. AIMET (★★★★)
   https://github.com/quic/aimet
   → 量化部署工具
```

### 7.3 数据集

```
训练数据集：
├─ DIV2K (800张高质量图像)
├─ Flickr2K (2650张高质量图像)
├─ ImageNet (大规模)
└─ RAISE (Raw 图像数据集)

测试数据集：
├─ Set5, Set14 (经典超分测试集)
├─ BSD100
├─ Urban100
└─ SIDD (真实噪声数据集)
```

### 7.4 在线资源

```
博客教程：
1. "Understanding Image Restoration" (Google AI Blog)
2. "Deep Learning for Image Super-Resolution" (Distill.pub)

视频课程：
1. Stanford CS231n (深度学习基础)
2. YouTube: "Image Restoration Explained"

社区：
1. Papers with Code
2. Reddit: r/computervision
3. 知乎专栏：计算机视觉
```

---

## 8. 总结与检验

### 8.1 学习路线回顾

```
Week 1: 图像质量评价指标
  └─ PSNR, SSIM, LPIPS

Week 2: ISP 流程与原理
  └─ Bayer, 去马赛克, 白平衡, Gamma

Week 3-4: 图像退化模型
  └─ 噪声, 模糊, 组合退化, Real-ESRGAN

Week 5: 感知损失与高级技术
  └─ Perceptual Loss, GAN

Week 6-8: 实战项目
  └─ 完整的 AI ISP 项目

时间：
- 全职学习：6-8 周
- 业余学习：3-4 个月
```

### 8.2 自我检验

**基础知识**：
```
□ 能解释 PSNR, SSIM, LPIPS 的区别
□ 能手写 ISP 的主要模块
□ 理解各种退化模型的原理
□ 理解感知损失的优势
```

**编程能力**：
```
□ 能实现完整的退化管道
□ 能搭建训练框架
□ 能复现经典论文代码
□ 能进行模型量化部署
```

**项目经验**：
```
□ 完成至少一个完整项目
□ 达到论文水平的指标
□ 能在真实数据上测试
□ 有量化部署经验
```

### 8.3 下一步建议

```
如果您已完成这份学习路线：

短期（1-2个月）：
├─ 深入一个方向（如 Transformer）
├─ 参加相关竞赛（如 NTIRE）
└─ 投稿开源项目

中期（3-6个月）：
├─ 发表技术博客/论文
├─ 开发自己的方法
└─ 积累实际项目经验

长期（1年+）：
├─ 成为领域专家
├─ 指导他人学习
└─ 推动技术创新
```

---

**祝您学习顺利！**

有任何问题，随时交流。从 High-level 到 Low-level 的转变需要时间，但凭借您8年的经验，相信您能快速掌握！💪

