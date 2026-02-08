# NAFNet 量化常见问题与解决方案

> **适用场景**：Low-level 视觉任务（去噪、去模糊、超分等）的模型量化部署  
> **模型**：NAFNet (Simple Baseline for Image Restoration)  
> **量化方案**：PTQ (Post-Training Quantization) 或 QAT (Quantization-Aware Training)

---

## 📋 目录

1. [问题概览](#问题概览)
2. [高光网格伪影](#1-高光网格伪影)
3. [高饱和区域鬼影](#2-高饱和区域鬼影ghost)
4. [暗区噪声放大](#3-暗区噪声放大)
5. [色彩偏移与色块](#4-色彩偏移与色块)
6. [边缘振铃效应](#5-边缘振铃效应ringing)
7. [时域不稳定](#6-时域不稳定temporal-instability)
8. [通用解决方案](#通用解决方案)
9. [代码实现](#代码实现)

---

## 问题概览

| 问题类型 | 严重程度 | 出现频率 | 主要原因 | 优先级 |
|---------|---------|---------|---------|--------|
| **高光网格** | ⭐⭐⭐⭐ | 高 | 激活量化饱和 | P0 |
| **高饱和鬼影** | ⭐⭐⭐⭐⭐ | 中 | 时域融合失效 | P0 |
| **暗区噪声** | ⭐⭐⭐ | 高 | 量化精度不足 | P1 |
| **色彩偏移** | ⭐⭐⭐⭐ | 中 | 通道量化不均 | P1 |
| **边缘振铃** | ⭐⭐⭐ | 低 | 高频丢失 | P2 |
| **时域抖动** | ⭐⭐⭐⭐ | 中 | 帧间不一致 | P1 |

---

## 1. 高光网格伪影

### 现象描述

**视觉表现**：
- 高亮区域出现 **规则的网格状图案**（grid pattern）
- 通常在接近饱和的高光边缘最明显
- 可能伴随 **块状伪影**（blockiness）

**典型场景**：
- 天空、白墙、高光反射
- 强光源边缘（路灯、阳光）
- 曝光过度区域

### 根因分析

#### 主要原因 1：激活量化饱和（Activation Clipping）

```
问题链路：
高光输入 (0.9-1.0) 
  ↓ NAFNet 处理
特征响应过大 (超出 int8 范围)
  ↓ 量化
Clip 到 max_val
  ↓ 解量化
所有高光值变成相同值
  ↓ 输出
网格状伪影
```

**数学解释**：

假设激活量化为 int8 [-128, 127]：
```python
scale = (max_val - min_val) / 255
quantized = clip(round(activation / scale), -128, 127)

# 当 activation >> scale * 127 时
# 所有值都被 clip 到 127
# 解量化后都是 scale * 127，丢失细节
```

#### 主要原因 2：Per-Tensor 量化粒度过粗

```
问题：
- NAFNet 的 SimpleGate 模块会产生很大的激活范围差异
- Per-Tensor 量化用同一个 scale 覆盖所有通道
- 高光区域的通道被"牺牲"

解决：
- 使用 Per-Channel 量化
```

#### 主要原因 3：网络结构特性

NAFNet 使用 **SimpleGate**：
```python
x1, x2 = x.chunk(2, dim=1)
out = x1 * x2  # 乘法会放大数值
```

高光区域 `x1 ≈ x2 ≈ 1.0`，乘法后仍为 1.0，但中间特征可能很大。

### 解决方案

#### ✅ 方案 1：分段量化（Piecewise Quantization）

**核心思想**：对高光区域使用更高精度或不同的量化参数。

```python
class PiecewiseQuantizer:
    """分段量化：高光区域特殊处理"""
    
    def __init__(self, threshold=0.8, low_bits=8, high_bits=16):
        self.threshold = threshold
        self.low_bits = low_bits
        self.high_bits = high_bits
    
    def quantize(self, x):
        # 分离高光和正常区域
        mask_highlight = x > self.threshold
        
        # 高光区域用高精度
        x_high = x[mask_highlight]
        scale_high = x_high.max() / (2 ** self.high_bits - 1)
        x_high_q = torch.round(x_high / scale_high).clamp(0, 2**self.high_bits - 1)
        
        # 正常区域用常规精度
        x_low = x[~mask_highlight]
        scale_low = x_low.max() / (2 ** self.low_bits - 1)
        x_low_q = torch.round(x_low / scale_low).clamp(0, 2**self.low_bits - 1)
        
        # 合并
        x_q = torch.zeros_like(x)
        x_q[mask_highlight] = x_high_q * scale_high
        x_q[~mask_highlight] = x_low_q * scale_low
        
        return x_q
```

#### ✅ 方案 2：激活范围预处理

**在量化前压缩高光范围**：

```python
def compress_highlights(x, gamma=2.2, threshold=0.8):
    """
    高光压缩，类似 Tone Mapping
    
    保持 [0, threshold] 线性
    [threshold, 1.0] 使用 gamma 压缩
    """
    x_compressed = x.clone()
    mask = x > threshold
    
    # 对高光部分应用 gamma 压缩
    x_high = x[mask]
    x_high_normalized = (x_high - threshold) / (1.0 - threshold)
    x_high_compressed = torch.pow(x_high_normalized, 1.0 / gamma)
    x_high_compressed = x_high_compressed * (1.0 - threshold) + threshold
    
    x_compressed[mask] = x_high_compressed
    return x_compressed

# 在网络输入前
x_compressed = compress_highlights(x)
output = model(x_compressed)
output = decompress_highlights(output)  # 反向操作
```

#### ✅ 方案 3：Per-Channel 激活量化

```python
import torch
from torch import nn

class PerChannelActivationQuantizer(nn.Module):
    """Per-Channel 激活量化"""
    
    def __init__(self, num_channels, bits=8):
        super().__init__()
        self.num_channels = num_channels
        self.bits = bits
        self.register_buffer('scales', torch.ones(num_channels))
        self.register_buffer('zero_points', torch.zeros(num_channels))
    
    def calibrate(self, x):
        """校准：计算每个通道的量化参数"""
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        
        for c in range(C):
            x_c = x[:, c, :, :].flatten()
            min_val = x_c.min()
            max_val = x_c.max()
            
            # 对称量化
            abs_max = max(abs(min_val), abs(max_val))
            self.scales[c] = abs_max / (2 ** (self.bits - 1) - 1)
            self.zero_points[c] = 0
    
    def forward(self, x):
        """量化 + 解量化"""
        B, C, H, W = x.shape
        x_q = torch.zeros_like(x)
        
        for c in range(C):
            scale = self.scales[c]
            zp = self.zero_points[c]
            
            x_c = x[:, c, :, :]
            x_c_q = torch.round(x_c / scale + zp)
            x_c_q = x_c_q.clamp(-2**(self.bits-1), 2**(self.bits-1) - 1)
            x_q[:, c, :, :] = (x_c_q - zp) * scale
        
        return x_q
```

#### ✅ 方案 4：混合精度（敏感层用 FP16）

```python
class NAFNetQuantized(nn.Module):
    def __init__(self, nafnet_model):
        super().__init__()
        self.model = nafnet_model
        
        # 标记敏感层（高光相关）
        self.sensitive_layers = [
            'encoder.0',  # 第一层编码器
            'decoder.-1',  # 最后一层解码器
            # 可以根据实际情况调整
        ]
    
    def forward(self, x):
        # 遍历所有层
        for name, module in self.model.named_modules():
            if name in self.sensitive_layers:
                # 敏感层使用 FP16
                module.to(torch.float16)
            else:
                # 其他层量化为 INT8
                quantize_module(module, bits=8)
        
        return self.model(x)
```

---

## 2. 高饱和区域鬼影（Ghost）

### 现象描述

**视觉表现**：
- **高饱和颜色区域**（深红、鲜蓝）出现 **拖影/重影**
- 视频场景下，运动物体后方有"残影"
- 京东快递员红色衣服是典型案例

**触发条件**：
- 视频帧间变化
- 高饱和 + 高对比度
- 运动场景

### 根因分析

#### 主要原因 1：时域融合（TNR）运动检测失效

```
问题链路：
高饱和区域 (Cb/Cr 极值)
  ↓ 量化
色度信息损失/抖动
  ↓ 运动检测
误判为"静止"（因为量化后色度相似）
  ↓ 时域融合
错误地融合历史帧
  ↓ 输出
鬼影/拖尾
```

**示例**：

```python
# 运动检测通常基于像素差异
diff = abs(frame_t - frame_t-1)
motion_mask = diff > threshold

# 量化后，高饱和区域的色度差异被抹平
# 导致 diff 很小，motion_mask = False
# 错误地认为"没有运动"，融合了历史帧
```

#### 主要原因 2：NAFNet 输出帧间抖动

量化导致网络输出在帧间不稳定（edge shimmer），即使输入相同：

```
Frame t:   Input → NAFNet_Q → Output_t
Frame t+1: Input → NAFNet_Q → Output_t+1

理论上 Input 相同时，Output_t == Output_t+1
但量化误差导致 Output_t ≠ Output_t+1

时域滤波器误以为是"运动"或"噪声"
```

#### 主要原因 3：色度通道量化不均

```
YCbCr 空间中：
- Y (亮度): 动态范围大，量化 scale 大
- Cb/Cr (色度): 动态范围小，但高饱和时有极值

Per-Tensor 量化时：
scale = max(Y_range, Cb_range, Cr_range) / 255
      ≈ Y_range / 255  (因为 Y >> Cb, Cr)

结果：Cb/Cr 的量化精度被严重压缩
```

### 解决方案

#### ✅ 方案 1：关闭/降低时域融合强度

**快速止血方案**：

```python
class TemporalFilterControl:
    """时域滤波控制：在高饱和区域降低融合"""
    
    def __init__(self, saturation_threshold=0.8):
        self.saturation_threshold = saturation_threshold
    
    def compute_saturation(self, img_rgb):
        """计算饱和度"""
        # RGB → HSV
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        saturation = img_hsv[:, :, 1] / 255.0
        return saturation
    
    def adaptive_tnr(self, frame_curr, frame_prev, alpha=0.3):
        """自适应时域降噪"""
        # 计算饱和度
        sat_curr = self.compute_saturation(frame_curr)
        
        # 高饱和区域降低融合强度
        mask_high_sat = sat_curr > self.saturation_threshold
        alpha_map = np.ones_like(sat_curr) * alpha
        alpha_map[mask_high_sat] = 0.1  # 高饱和区域几乎不融合
        
        # 自适应融合
        frame_out = alpha_map[:, :, None] * frame_curr + \
                    (1 - alpha_map[:, :, None]) * frame_prev
        
        return frame_out
```

#### ✅ 方案 2：改进运动检测（考虑量化误差）

```python
class RobustMotionDetector:
    """鲁棒运动检测：容忍量化噪声"""
    
    def __init__(self, threshold=10.0, quant_noise_std=2.0):
        self.threshold = threshold
        self.quant_noise_std = quant_noise_std
    
    def detect_motion(self, frame_t, frame_t1, quantized=True):
        """
        运动检测，考虑量化噪声
        
        Args:
            quantized: 是否来自量化模型
        """
        # 像素差异
        diff = np.abs(frame_t.astype(float) - frame_t1.astype(float))
        
        if quantized:
            # 量化模型需要更高的阈值
            # 因为存在量化噪声导致的"假动"
            effective_threshold = self.threshold + 2 * self.quant_noise_std
        else:
            effective_threshold = self.threshold
        
        motion_mask = diff > effective_threshold
        
        # 形态学处理：去除孤立噪点
        kernel = np.ones((3, 3), np.uint8)
        motion_mask = cv2.morphologyEx(
            motion_mask.astype(np.uint8), 
            cv2.MORPH_OPEN, 
            kernel
        )
        
        return motion_mask.astype(bool)
```

#### ✅ 方案 3：QAT 加入时域一致性损失

```python
import torch
import torch.nn as nn

class TemporalConsistencyLoss(nn.Module):
    """时域一致性损失：减少帧间抖动"""
    
    def __init__(self, weight=0.1):
        super().__init__()
        self.weight = weight
    
    def forward(self, output_t, output_t1, flow=None):
        """
        Args:
            output_t: 当前帧输出
            output_t1: 前一帧输出
            flow: 光流（可选，用于运动补偿）
        """
        if flow is not None:
            # 使用光流对齐 output_t1
            output_t1_warped = self.warp_flow(output_t1, flow)
        else:
            output_t1_warped = output_t1
        
        # 时域差异
        temporal_diff = torch.abs(output_t - output_t1_warped)
        
        # 损失
        loss = temporal_diff.mean()
        
        return self.weight * loss
    
    def warp_flow(self, img, flow):
        """使用光流warp图像"""
        # 使用 grid_sample 实现
        B, C, H, W = img.shape
        
        # 创建网格
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=img.device),
            torch.arange(W, device=img.device)
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).float()
        
        # 添加光流
        grid = grid + flow.permute(0, 2, 3, 1)
        
        # 归一化到 [-1, 1]
        grid[:, :, :, 0] = 2.0 * grid[:, :, :, 0] / (W - 1) - 1.0
        grid[:, :, :, 1] = 2.0 * grid[:, :, :, 1] / (H - 1) - 1.0
        
        # Warp
        img_warped = torch.nn.functional.grid_sample(
            img, grid, align_corners=True
        )
        
        return img_warped

# 在 QAT 训练中使用
tc_loss = TemporalConsistencyLoss(weight=0.1)
total_loss = reconstruction_loss + tc_loss(output_t, output_t1, flow)
```

#### ✅ 方案 4：色度通道独立量化

```python
class ChromaAwareQuantizer:
    """色度感知量化：YCbCr 各通道独立"""
    
    def __init__(self, y_bits=8, cbcr_bits=8):
        self.y_bits = y_bits
        self.cbcr_bits = cbcr_bits
    
    def quantize_ycbcr(self, img_rgb):
        """
        RGB → YCbCr → 独立量化 → RGB
        """
        # RGB → YCbCr
        img_ycbcr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
        
        y, cb, cr = img_ycbcr[:, :, 0], img_ycbcr[:, :, 1], img_ycbcr[:, :, 2]
        
        # Y 通道量化
        y_scale = 255.0 / (2 ** self.y_bits - 1)
        y_q = np.round(y / y_scale) * y_scale
        
        # Cb/Cr 通道量化（独立 scale）
        cb_scale = (cb.max() - cb.min()) / (2 ** self.cbcr_bits - 1)
        cr_scale = (cr.max() - cr.min()) / (2 ** self.cbcr_bits - 1)
        
        cb_q = np.round((cb - cb.min()) / cb_scale) * cb_scale + cb.min()
        cr_q = np.round((cr - cr.min()) / cr_scale) * cr_scale + cr.min()
        
        # 合并
        img_ycbcr_q = np.stack([y_q, cb_q, cr_q], axis=-1).astype(np.uint8)
        
        # YCbCr → RGB
        img_rgb_q = cv2.cvtColor(img_ycbcr_q, cv2.COLOR_YCrCb2RGB)
        
        return img_rgb_q
```

---

## 3. 暗区噪声放大

### 现象描述

**视觉表现**：
- **暗部区域**出现明显的颗粒噪声
- 原本平滑的暗部变得粗糙
- 量化后的暗区噪声 > 量化前

**典型场景**：
- 夜景、阴影
- 室内暗光
- 低曝光区域

### 根因分析

#### 主要原因 1：量化步长过大

```
暗区值范围：[0, 0.1]
量化 scale (全局)：1.0 / 255 ≈ 0.004

可用的量化级别：0.1 / 0.004 ≈ 25 级
原始 FP32：无限精度

结果：暗区被量化为 25 个离散值
表现为"台阶"和噪声
```

#### 主要原因 2：去噪网络的噪声放大

```
NAFNet 去噪原理：
- 学习 residual: noise = input - clean
- 输出: output = input - noise

量化误差：
- noise_q = noise + quant_error
- output_q = input - noise_q
         = input - noise - quant_error
         = clean - quant_error

在暗区，quant_error 相对幅度更大
```

#### 主要原因 3：非线性层的误差累积

暗区信号本身很小，经过多层非线性（ReLU, SimpleGate）后，量化误差会累积放大。

### 解决方案

#### ✅ 方案 1：对数量化（Logarithmic Quantization）

模拟人眼感知，对暗部使用更细的量化：

```python
class LogQuantizer:
    """对数量化：暗部细、亮部粗"""
    
    def __init__(self, bits=8, epsilon=1e-6):
        self.bits = bits
        self.epsilon = epsilon
        self.max_val = 2 ** bits - 1
    
    def quantize(self, x):
        """
        x: [0, 1]
        """
        # 对数映射
        x_log = np.log(x + self.epsilon)
        
        # 量化
        x_log_min = np.log(self.epsilon)
        x_log_max = np.log(1.0 + self.epsilon)
        
        x_log_norm = (x_log - x_log_min) / (x_log_max - x_log_min)
        x_q = np.round(x_log_norm * self.max_val)
        
        # 反量化
        x_log_dq = x_q / self.max_val * (x_log_max - x_log_min) + x_log_min
        x_dq = np.exp(x_log_dq) - self.epsilon
        
        return np.clip(x_dq, 0, 1)
```

#### ✅ 方案 2：暗区使用更高精度

```python
class AdaptiveBitwidthQuantizer:
    """自适应位宽：暗区 16-bit，亮区 8-bit"""
    
    def __init__(self, dark_threshold=0.2):
        self.dark_threshold = dark_threshold
    
    def quantize(self, x):
        # 分离暗区和亮区
        mask_dark = x < self.dark_threshold
        
        x_q = np.zeros_like(x)
        
        # 暗区：16-bit
        x_dark = x[mask_dark]
        scale_dark = 1.0 / (2 ** 16 - 1)
        x_dark_q = np.round(x_dark / scale_dark) * scale_dark
        x_q[mask_dark] = x_dark_q
        
        # 亮区：8-bit
        x_bright = x[~mask_dark]
        scale_bright = 1.0 / (2 ** 8 - 1)
        x_bright_q = np.round(x_bright / scale_bright) * scale_bright
        x_q[~mask_dark] = x_bright_q
        
        return x_q
```

#### ✅ 方案 3：Bias Correction（偏置校正）

量化后，暗区的均值可能漂移，需要校正：

```python
class BiasCorrection:
    """偏置校正：修正量化后的统计偏移"""
    
    def __init__(self):
        self.bias_map = {}
    
    def calibrate(self, x_fp32, x_quant, layer_name):
        """
        校准：计算偏置
        
        Args:
            x_fp32: FP32 激活
            x_quant: 量化后激活
        """
        # 计算误差均值（按通道）
        bias = (x_fp32 - x_quant).mean(dim=(0, 2, 3))
        self.bias_map[layer_name] = bias
    
    def correct(self, x_quant, layer_name):
        """校正"""
        if layer_name in self.bias_map:
            bias = self.bias_map[layer_name]
            x_corrected = x_quant + bias.view(1, -1, 1, 1)
            return x_corrected
        return x_quant
```

#### ✅ 方案 4：QAT 加入暗区损失

```python
class DarkRegionLoss(nn.Module):
    """暗区损失：加强暗部监督"""
    
    def __init__(self, threshold=0.2, weight=2.0):
        super().__init__()
        self.threshold = threshold
        self.weight = weight
    
    def forward(self, output, target):
        # 暗区 mask
        mask_dark = (target < self.threshold).float()
        
        # 暗区损失
        loss_dark = ((output - target) ** 2 * mask_dark).sum() / (mask_dark.sum() + 1e-6)
        
        # 全局损失
        loss_global = ((output - target) ** 2).mean()
        
        return loss_global + self.weight * loss_dark
```

---

## 4. 色彩偏移与色块

### 现象描述

**视觉表现**：
- 整体色调偏移（偏红/偏蓝/偏绿）
- 平滑渐变区域出现 **色块**（color banding）
- 色彩饱和度异常

### 根因分析

#### 主要原因：RGB 通道量化不均

```
问题：
- Per-Tensor 量化：scale = max(R, G, B) / 255
- 如果某个通道动态范围特别大（如天空的蓝色）
- 其他通道的量化精度会被牺牲

结果：
- 通道间量化误差不平衡 → 色偏
- 某些通道量化级数太少 → 色块
```

### 解决方案

#### ✅ 方案：Per-Channel 量化 + 色彩空间转换

```python
class ColorPreservingQuantizer:
    """保色量化"""
    
    def quantize_rgb(self, img_rgb, bits=8):
        """RGB 各通道独立量化"""
        img_q = np.zeros_like(img_rgb)
        
        for c in range(3):
            channel = img_rgb[:, :, c]
            scale = channel.max() / (2 ** bits - 1)
            channel_q = np.round(channel / scale) * scale
            img_q[:, :, c] = channel_q
        
        return img_q
    
    def quantize_yuv(self, img_rgb, y_bits=8, uv_bits=6):
        """
        YUV 量化：
        - Y 通道高精度（亮度敏感）
        - UV 通道可降低精度（色度不敏感）
        """
        # RGB → YUV
        img_yuv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)
        
        # Y 量化
        y = img_yuv[:, :, 0]
        y_scale = 255.0 / (2 ** y_bits - 1)
        y_q = np.round(y / y_scale) * y_scale
        
        # UV 量化（粗量化）
        u = img_yuv[:, :, 1]
        v = img_yuv[:, :, 2]
        uv_scale = 255.0 / (2 ** uv_bits - 1)
        u_q = np.round(u / uv_scale) * uv_scale
        v_q = np.round(v / uv_scale) * uv_scale
        
        # 合并
        img_yuv_q = np.stack([y_q, u_q, v_q], axis=-1).astype(np.uint8)
        
        # YUV → RGB
        img_rgb_q = cv2.cvtColor(img_yuv_q, cv2.COLOR_YUV2RGB)
        
        return img_rgb_q
```

---

## 5. 边缘振铃效应（Ringing）

### 现象描述

**视觉表现**：
- 强边缘附近出现 **波纹/振荡**
- 类似 JPEG 压缩伪影

### 根因分析

量化导致高频细节丢失，网络试图重建时产生过冲/下冲。

### 解决方案

```python
class EdgePreservingQuantizer:
    """边缘保持量化"""
    
    def quantize_with_edge_detection(self, img, bits=8):
        # 检测边缘
        edges = cv2.Canny((img * 255).astype(np.uint8), 50, 150)
        
        # 边缘区域使用更高精度
        # （实现略）
        pass
```

---

## 6. 时域不稳定（Temporal Instability）

### 现象描述

**视觉表现**：
- 视频出现 **闪烁/抖动**
- 静止场景下输出仍在变化

### 解决方案

参考 [方案 2.3](#✅-方案-3qat-加入时域一致性损失)

---

## 通用解决方案

### 1. 量化感知训练（QAT）

```python
import torch
import torch.quantization as quant

class NAFNetQAT:
    """NAFNet 量化感知训练"""
    
    def __init__(self, model):
        self.model = model
        self.prepare_qat()
    
    def prepare_qat(self):
        """准备 QAT"""
        # 融合层
        torch.quantization.fuse_modules(
            self.model, 
            [['conv', 'bn', 'relu']], 
            inplace=True
        )
        
        # 设置量化配置
        self.model.qconfig = quant.get_default_qat_qconfig('fbgemm')
        
        # 准备
        quant.prepare_qat(self.model, inplace=True)
    
    def train(self, dataloader, epochs=10):
        """QAT 训练"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        
        for epoch in range(epochs):
            for batch in dataloader:
                img_input, img_target = batch
                
                # 前向
                output = self.model(img_input)
                
                # 损失（加入多种损失）
                loss_recon = F.mse_loss(output, img_target)
                loss_dark = DarkRegionLoss()(output, img_target)
                loss_temporal = TemporalConsistencyLoss()(output, prev_output)
                
                loss = loss_recon + loss_dark + loss_temporal
                
                # 反向
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
    def convert_to_quantized(self):
        """转换为量化模型"""
        self.model.eval()
        quant.convert(self.model, inplace=True)
        return self.model
```

### 2. 校准数据集构建

```python
class CalibrationDataset:
    """校准数据集：覆盖极端场景"""
    
    def __init__(self):
        self.samples = []
    
    def add_corner_cases(self):
        """添加边角案例"""
        # 1. 高光场景
        self.samples.append(self.create_highlight_image())
        
        # 2. 暗光场景
        self.samples.append(self.create_dark_image())
        
        # 3. 高饱和场景
        self.samples.append(self.create_saturated_image())
        
        # 4. 高对比度场景
        self.samples.append(self.create_high_contrast_image())
    
    def create_highlight_image(self):
        """生成高光测试图"""
        img = np.ones((512, 512, 3)) * 0.95
        # 添加一些细节
        img[100:200, 100:200] = 1.0
        return img
    
    def create_saturated_image(self):
        """生成高饱和测试图"""
        img = np.zeros((512, 512, 3))
        img[:, :, 0] = 1.0  # 纯红
        return img
```

---

## 代码实现

### 完整的量化Pipeline

```python
import torch
import torch.nn as nn
import numpy as np

class NAFNetQuantizationPipeline:
    """NAFNet 量化完整流程"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
        # 初始化各种量化器
        self.piecewise_quantizer = PiecewiseQuantizer()
        self.perchannel_quantizer = PerChannelActivationQuantizer(
            num_channels=config.num_channels
        )
        self.log_quantizer = LogQuantizer()
        self.bias_corrector = BiasCorrection()
        
    def quantize_model(self, calibration_data):
        """
        量化模型
        
        Args:
            calibration_data: 校准数据
        """
        print("Step 1: 校准量化参数...")
        self.calibrate(calibration_data)
        
        print("Step 2: 应用量化...")
        self.apply_quantization()
        
        print("Step 3: 偏置校正...")
        self.apply_bias_correction(calibration_data)
        
        print("Step 4: 验证...")
        self.validate()
        
        return self.model
    
    def calibrate(self, data):
        """校准"""
        self.model.eval()
        with torch.no_grad():
            for batch in data:
                # 运行模型，收集激活统计
                _ = self.model(batch)
    
    def apply_quantization(self):
        """应用量化"""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                # 量化权重
                self.quantize_weights(module)
            
            if hasattr(module, 'activation'):
                # 量化激活
                self.quantize_activations(module, name)
    
    def quantize_weights(self, module):
        """量化权重（Per-Channel）"""
        weight = module.weight.data
        C_out, C_in, K, K = weight.shape
        
        # Per-Channel 量化
        for c in range(C_out):
            w_c = weight[c, :, :, :]
            scale = w_c.abs().max() / 127
            w_c_q = torch.round(w_c / scale).clamp(-128, 127) * scale
            weight[c, :, :, :] = w_c_q
        
        module.weight.data = weight
    
    def quantize_activations(self, module, name):
        """量化激活"""
        # 根据层类型选择量化策略
        if 'highlight_sensitive' in name:
            # 高光敏感层：分段量化
            module.quantizer = self.piecewise_quantizer
        elif 'dark_sensitive' in name:
            # 暗部敏感层：对数量化
            module.quantizer = self.log_quantizer
        else:
            # 普通层：Per-Channel 量化
            module.quantizer = self.perchannel_quantizer
    
    def apply_bias_correction(self, data):
        """偏置校正"""
        # 收集 FP32 和量化后的激活
        # 计算偏置并应用
        pass
    
    def validate(self):
        """验证量化质量"""
        # 检查各种指标
        pass

# 使用示例
if __name__ == '__main__':
    # 1. 加载模型
    model = NAFNet()
    
    # 2. 准备配置
    config = {
        'num_channels': 64,
        'bits': 8,
        'calibration_samples': 100
    }
    
    # 3. 准备校准数据
    calib_dataset = CalibrationDataset()
    calib_dataset.add_corner_cases()
    
    # 4. 量化
    pipeline = NAFNetQuantizationPipeline(model, config)
    model_quantized = pipeline.quantize_model(calib_dataset)
    
    # 5. 保存
    torch.save(model_quantized.state_dict(), 'nafnet_quantized.pth')
```

---

## 总结

### 关键要点

1. ✅ **高光网格** → 分段量化 + Per-Channel
2. ✅ **高饱和鬼影** → 降低TNR + 时域一致性
3. ✅ **暗区噪声** → 对数量化 + 偏置校正
4. ✅ **色彩偏移** → Per-Channel + 色彩空间
5. ✅ **通用方案** → QAT + 多损失函数

### 推荐工作流

```
FP32 模型
  ↓
PTQ (快速验证)
  ↓
问题诊断（本文档）
  ↓
针对性修复
  ↓
QAT (精调)
  ↓
部署验证
```

---

*最后更新：2026-02-08*  
*适用于：NAFNet, Restormer, SwinIR 等 Restoration 网络*

