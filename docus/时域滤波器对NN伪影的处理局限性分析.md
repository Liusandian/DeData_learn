# 时域滤波器对NN伪影的处理局限性分析

> **问题**：为什么 MATF (Motion Adaptive Temporal Filter) 能有效处理随机噪声，却难以去除规律性 Pattern 和细网格伪影？  
> **场景**：NN 网络推理输出 + Vanilla 画质仿真（本地C代码）

---

## 📋 问题概述

### 典型场景

```
NN 推理输出（带伪影）
  ↓
MATF / TNR（时域滤波）
  ↓
结果：
✅ 随机噪声 → 有效抑制
❌ 规律网格 → 几乎无效
❌ Pattern 伪影 → 难以去除
```

### 两类伪影对比

| 特性 | 随机噪声 | 规律 Pattern/网格 |
|------|----------|------------------|
| **空间特性** | 无序、白噪声 | 有序、周期性 |
| **时域特性** | 帧间独立 | 帧间稳定/相干 |
| **频率特性** | 宽频、高频 | 窄频、特定频率 |
| **MATF 效果** | ✅ 显著 | ❌ 微弱 |
| **根本原因** | 时域平均有效 | 时域平均无效 |

---

## 🔬 深度分析：为什么随机噪声可以被 MATF 去除？

### 1. MATF 工作原理

**运动自适应时域滤波器（MATF）**：

```c
// 伪代码
float matf_filter(float curr_pixel, float prev_pixel, float motion) {
    // 运动检测
    float alpha = compute_alpha(motion);  // motion 越大，alpha 越大
    
    // 时域融合
    float output = alpha * curr_pixel + (1 - alpha) * prev_pixel;
    
    return output;
}

// alpha ∈ [0, 1]
// motion 小 → alpha 小 → 更多融合历史帧（降噪）
// motion 大 → alpha 大 → 更少融合（保留细节）
```

**核心思想**：
- 静止区域：多帧平均 → 噪声被平滑
- 运动区域：减少融合 → 避免拖影

### 2. 随机噪声的特性

#### 时域独立性

```
帧 t:   signal + noise_t
帧 t+1: signal + noise_t+1
帧 t+2: signal + noise_t+2

其中：
- signal 在静止区域是恒定的
- noise_t, noise_t+1, noise_t+2 相互独立

时域平均：
mean = (signal + noise_t + signal + noise_t+1 + ... + signal + noise_t+N) / N
     = signal + (noise_t + noise_t+1 + ... + noise_t+N) / N
     → signal + 0  (噪声均值趋向0)
```

#### 数学证明

对于零均值高斯噪声 $n \sim \mathcal{N}(0, \sigma^2)$：

$$
\text{Var}(\frac{1}{N}\sum_{i=1}^N n_i) = \frac{1}{N^2} \sum_{i=1}^N \text{Var}(n_i) = \frac{\sigma^2}{N}
$$

**结论**：N 帧平均后，噪声方差降低为原来的 1/N。

#### 频域特性

随机噪声的功率谱密度（PSD）：

```
P(f) = σ² (白噪声，所有频率能量均匀)
```

时域滤波器在频域等价于低通滤波：

```
H(f) = α + (1-α) * e^(-j2πf)  (简化模型)
```

对于高频噪声，`H(f)` 衰减明显。

### 3. 实际效果

```python
# 模拟随机噪声 + MATF
import numpy as np

signal = 0.5  # 恒定信号
noise_std = 0.1
N = 10  # 平均 10 帧

frames = [signal + np.random.randn() * noise_std for _ in range(N)]
filtered = np.mean(frames)

print(f"原始信号: {signal}")
print(f"单帧噪声: {frames[0]:.3f}")
print(f"滤波后: {filtered:.3f}")
# 输出：滤波后 ≈ 0.500，噪声被有效抑制
```

---

## ❌ 为什么规律 Pattern/网格难以去除？

### 1. 规律 Pattern 的特性

#### 时域稳定性（Temporal Coherence）

```
帧 t:   signal + pattern
帧 t+1: signal + pattern  (相同的 pattern)
帧 t+2: signal + pattern

时域平均：
mean = (signal + pattern + signal + pattern + ... ) / N
     = signal + pattern  (pattern 没有被平滑！)
```

**关键差异**：Pattern 在帧间是**相干的**（coherent），不满足独立性假设。

#### 空间周期性

规律网格通常有固定周期（如 8×8、16×16）：

```
Pattern(x, y) = A * sin(2π * x / T_x) * sin(2π * y / T_y)

其中 T_x, T_y 是周期
```

这种周期性在时域上也是稳定的：

```
Pattern_t(x, y) ≈ Pattern_{t+1}(x, y)
```

### 2. MATF 失效的数学原理

#### 时域滤波器的局限

MATF 的输出：

$$
O_t = \alpha \cdot I_t + (1-\alpha) \cdot O_{t-1}
$$

展开（假设静止场景，$\alpha$ 很小）：

$$
O_t = \alpha \sum_{k=0}^{\infty} (1-\alpha)^k I_{t-k}
$$

对于随机噪声 $n_t$：
$$
E[n_t \cdot n_{t-k}] = 0, \quad k \neq 0 \quad \text{(不相关)}
$$

对于规律 Pattern $p_t$：
$$
p_t = p_{t-k}, \quad \forall k \quad \text{(完全相关)}
$$

因此：
$$
O_t^{\text{pattern}} = \alpha \sum_{k=0}^{\infty} (1-\alpha)^k p = p \quad \text{(Pattern 保留)}
$$

#### 频域视角

```
随机噪声：
- 宽频（所有频率都有）
- MATF 低通滤波 → 高频被抑制 → 噪声降低

规律网格：
- 窄频（只在特定频率 f_0）
- MATF 对 f_0 的抑制很弱（如果 f_0 < 截止频率）
- Pattern 能量集中，难以衰减
```

### 3. 运动检测的无效性

MATF 的运动检测通常基于像素差异：

```c
float motion = abs(curr_pixel - prev_pixel);
```

对于规律 Pattern：
- 静止场景：`curr_pixel - prev_pixel ≈ 0`（Pattern 稳定）
- MATF 判定为"静止"
- 进行时域融合
- 但融合无效（因为 Pattern 在历史帧中也存在）

### 4. 典型案例分析

#### 案例 1：量化网格（8×8 块）

```
原因：
- NN 量化后，每个 8×8 块内部值接近
- 块边界有台阶
- 这种台阶在时域上是稳定的

MATF 效果：
- 运动检测：块内差异小 → 判定静止
- 时域融合：历史帧也有相同网格 → 融合无效
- 结果：网格保留
```

#### 案例 2：高光网格

```
原因：
- 激活量化饱和
- 高光区域形成规律的"平台"
- 平台在时域稳定

MATF 效果：
- 平台区域像素值恒定 → 运动 = 0
- 大量融合历史帧 → 但历史帧也是平台
- 结果：网格依然存在
```

---

## 🛠️ 解决方案

### 方案 1：空域滤波 + 边缘保持

**核心思想**：在空间域检测和消除 Pattern。

#### 1.1 双边滤波器（Bilateral Filter）

```c
// 双边滤波：保边缘去网格
float bilateral_filter(Image img, int x, int y, 
                       float sigma_spatial, float sigma_range) {
    float sum = 0.0f;
    float weight_sum = 0.0f;
    int radius = 3 * sigma_spatial;
    
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            int nx = x + dx;
            int ny = y + dy;
            
            // 空间权重
            float spatial_dist = sqrt(dx*dx + dy*dy);
            float w_spatial = exp(-spatial_dist * spatial_dist / 
                                  (2 * sigma_spatial * sigma_spatial));
            
            // 值域权重（保护边缘）
            float range_dist = img[ny][nx] - img[y][x];
            float w_range = exp(-range_dist * range_dist / 
                               (2 * sigma_range * sigma_range));
            
            float w = w_spatial * w_range;
            sum += w * img[ny][nx];
            weight_sum += w;
        }
    }
    
    return sum / weight_sum;
}

// 使用
sigma_spatial = 3.0;  // 空间范围
sigma_range = 0.1;    // 值域范围（小 → 强保边缘）
filtered = bilateral_filter(img, x, y, sigma_spatial, sigma_range);
```

**优点**：
- 去除平滑区域的网格
- 保留真实边缘

**局限**：
- 计算量大
- 参数敏感

#### 1.2 引导滤波器（Guided Filter）

```c
// 引导滤波：快速边缘保持
// 算法：He et al., ECCV 2010
void guided_filter(Image img, Image guide, Image out, int radius, float eps) {
    // 1. 计算局部均值
    mean_I = box_filter(guide, radius);
    mean_p = box_filter(img, radius);
    
    // 2. 计算局部协方差
    corr_I = box_filter(guide * guide, radius);
    corr_Ip = box_filter(guide * img, radius);
    
    var_I = corr_I - mean_I * mean_I;
    cov_Ip = corr_Ip - mean_I * mean_p;
    
    // 3. 计算线性系数
    a = cov_Ip / (var_I + eps);  // eps 控制保边缘强度
    b = mean_p - a * mean_I;
    
    // 4. 输出
    mean_a = box_filter(a, radius);
    mean_b = box_filter(b, radius);
    
    out = mean_a * guide + mean_b;
}

// 使用
guided_filter(noisy_img, guide_img, output, radius=4, eps=0.01);
// eps 小 → 强保边缘，网格去除少
// eps 大 → 弱保边缘，网格去除多
```

**优点**：
- 速度快（线性复杂度）
- 保边缘好
- 适合实时处理

### 方案 2：频域滤波（针对周期性 Pattern）

#### 2.1 陷波滤波器（Notch Filter）

```python
import numpy as np
from scipy import fft

def notch_filter_remove_grid(img, grid_freq_x, grid_freq_y, radius=5):
    """
    陷波滤波：去除特定频率的网格
    
    Args:
        img: 输入图像
        grid_freq_x: 网格在 x 方向的频率（如 1/8）
        grid_freq_y: 网格在 y 方向的频率（如 1/8）
        radius: 陷波半径
    """
    H, W = img.shape
    
    # 1. FFT
    img_fft = fft.fft2(img)
    img_fft_shift = fft.fftshift(img_fft)
    
    # 2. 创建陷波滤波器
    mask = np.ones((H, W))
    cy, cx = H // 2, W // 2
    
    # 计算网格频率对应的位置
    freq_x_pos = int(grid_freq_x * W)
    freq_y_pos = int(grid_freq_y * H)
    
    # 在频谱中"挖洞"（陷波）
    for dy in range(-radius, radius+1):
        for dx in range(-radius, radius+1):
            if dx*dx + dy*dy <= radius*radius:
                # 基频及谐波
                for k in range(1, 4):  # 考虑前 3 个谐波
                    mask[cy + k*freq_y_pos + dy, cx + k*freq_x_pos + dx] = 0
                    mask[cy - k*freq_y_pos + dy, cx - k*freq_x_pos + dx] = 0
                    mask[cy + k*freq_y_pos + dy, cx - k*freq_x_pos + dx] = 0
                    mask[cy - k*freq_y_pos + dy, cx + k*freq_x_pos + dx] = 0
    
    # 3. 应用滤波器
    img_fft_filtered = img_fft_shift * mask
    
    # 4. IFFT
    img_fft_ishift = fft.ifftshift(img_fft_filtered)
    img_filtered = fft.ifft2(img_fft_ishift).real
    
    return img_filtered

# 使用
# 假设 8×8 网格
img_clean = notch_filter_remove_grid(img, grid_freq_x=1/8, grid_freq_y=1/8)
```

**优点**：
- 针对性强（只去除特定频率）
- 不影响其他频率（保留细节）

**局限**：
- 需要预先知道网格频率
- FFT 计算量较大

#### 2.2 自适应频率检测 + 陷波

```python
def auto_detect_grid_frequency(img):
    """自动检测网格频率"""
    # 1. FFT
    img_fft = np.fft.fft2(img)
    magnitude = np.abs(np.fft.fftshift(img_fft))
    
    # 2. 排除低频（DC 及附近）
    H, W = img.shape
    cy, cx = H // 2, W // 2
    magnitude[cy-10:cy+10, cx-10:cx+10] = 0
    
    # 3. 寻找峰值（对应网格频率）
    peaks = []
    threshold = np.percentile(magnitude, 99.5)
    
    for y in range(H):
        for x in range(W):
            if magnitude[y, x] > threshold:
                peaks.append((y - cy, x - cx))
    
    # 4. 聚类找主频
    # （实现略，可用 DBSCAN）
    
    return peaks

# 使用
grid_freqs = auto_detect_grid_frequency(img)
for freq_y, freq_x in grid_freqs:
    img = notch_filter_remove_grid(img, freq_x/W, freq_y/H)
```

### 方案 3：混合方案（空域 + 时域）

#### 3.1 时空联合滤波

```c
typedef struct {
    float spatial_sigma;
    float temporal_alpha;
    float pattern_threshold;
} HybridFilterParams;

float hybrid_filter(Image curr, Image prev, int x, int y, 
                    HybridFilterParams params) {
    // 1. 检测是否为 Pattern 区域
    float variance = compute_local_variance(curr, x, y, radius=3);
    bool is_pattern = (variance < params.pattern_threshold);
    
    if (is_pattern) {
        // Pattern 区域：使用空域滤波（双边）
        return bilateral_filter(curr, x, y, 
                               params.spatial_sigma, 0.1);
    } else {
        // 正常区域：使用时域滤波（MATF）
        float motion = abs(curr[y][x] - prev[y][x]);
        float alpha = motion / 255.0;  // 简化
        return alpha * curr[y][x] + (1 - alpha) * prev[y][x];
    }
}
```

**思路**：
- 平滑区域（可能有网格）→ 空域滤波
- 纹理区域（真实细节）→ 时域滤波

#### 3.2 方差自适应混合

```c
float adaptive_hybrid_filter(Image curr, Image prev, int x, int y) {
    // 计算局部方差
    float var_curr = local_variance(curr, x, y);
    
    // 计算时域稳定性
    float temporal_var = abs(curr[y][x] - prev[y][x]);
    
    // 决策
    if (var_curr < 10.0 && temporal_var < 2.0) {
        // 低方差 + 时域稳定 = 可能是 Pattern
        // 使用空域滤波
        return guided_filter_pixel(curr, x, y);
    } else {
        // 正常区域：MATF
        return matf_filter(curr, prev, x, y);
    }
}
```

### 方案 4：机器学习方案

#### 4.1 轻量级去网格网络

```python
import torch
import torch.nn as nn

class GridRemovalNet(nn.Module):
    """轻量级去网格网络"""
    
    def __init__(self):
        super().__init__()
        # 编码器：检测网格
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU()
        )
        
        # 解码器：重建
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1)
        )
    
    def forward(self, x):
        # 编码
        feat = self.encoder(x)
        
        # 解码
        residual = self.decoder(feat)
        
        # 输出：原图 - 网格
        return x - residual  # residual 是预测的网格

# 训练
model = GridRemovalNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for img_with_grid, img_clean in dataloader:
    output = model(img_with_grid)
    loss = nn.MSELoss()(output, img_clean)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

#### 4.2 部署为 C 代码

训练后转换为 ONNX，再用 C 推理：

```bash
# 导出 ONNX
torch.onnx.export(model, dummy_input, "grid_removal.onnx")

# 使用 ONNX Runtime (C API) 推理
```

---

## 📊 方案对比

| 方案 | 计算量 | 效果 | 实时性 | 通用性 |
|------|--------|------|--------|--------|
| **双边滤波** | 中 | 中 | 中 | 高 |
| **引导滤波** | 低 | 中-高 | 高 ⭐ | 高 |
| **陷波滤波** | 高 | 高 | 低 | 低（需知频率）|
| **混合滤波** | 中 | 高 | 中 | 高 ⭐ |
| **深度学习** | 低（推理） | 高 | 中 | 中（需训练）|

---

## 🎯 推荐方案

### 场景 1：实时视频处理（Vanilla C 代码）

```
推荐：引导滤波 + MATF 混合

流程：
1. 检测平滑区域（方差 < 阈值）
2. 平滑区域 → 引导滤波（去网格）
3. 其他区域 → MATF（去噪声）
```

**代码框架**：

```c
void process_frame(Image curr, Image prev, Image out) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float var = local_variance(curr, x, y, 3);
            
            if (var < 10.0) {
                // 可能是网格区域：空域滤波
                out[y][x] = guided_filter_pixel(curr, x, y, 4, 0.01);
            } else {
                // 正常区域：时域滤波
                float motion = abs(curr[y][x] - prev[y][x]);
                float alpha = min(motion / 20.0, 1.0);
                out[y][x] = alpha * curr[y][x] + (1-alpha) * prev[y][x];
            }
        }
    }
}
```

### 场景 2：离线高质量处理

```
推荐：自动检测 + 陷波滤波

流程：
1. FFT 检测网格频率
2. 设计陷波滤波器
3. 频域去除网格
4. 可选：后续空域精修
```

### 场景 3：端侧部署

```
推荐：轻量级 CNN（转 ONNX/TFLite）

优势：
- 计算量可控（1-2ms）
- 效果好
- 一次训练，处处部署
```

---

## 🔬 根本原因总结

### 为什么 MATF 对随机噪声有效？

```
1. 时域独立性
   noise_t ⊥ noise_{t+1}  (统计独立)
   
2. 时域平均
   E[noise] = 0
   Var[mean(noise)] = σ²/N → 0
   
3. 频域视角
   白噪声 = 宽频
   MATF = 低通滤波器
   → 高频噪声被抑制
```

### 为什么 MATF 对规律 Pattern 无效？

```
1. 时域相干性
   pattern_t = pattern_{t+1}  (完全相同)
   
2. 时域平均无效
   E[pattern] = pattern
   平均不改变 pattern
   
3. 频域视角
   Pattern = 窄频（单一频率 f_0）
   MATF 对 f_0 抑制弱
   → Pattern 能量保留
```

### 本质差异

```
随机噪声：
- 时间域：独立 → 平均有效
- 频率域：宽谱 → 低通滤波有效

规律 Pattern：
- 时间域：相干 → 平均无效
- 频率域：窄谱 → 低通滤波无效（如果 f_0 在通带内）

结论：需要空域或特定频域方法
```

---

## 💡 实践建议

1. **快速验证**：先用引导滤波测试效果
2. **参数调优**：根据网格尺寸调整滤波器半径
3. **分区处理**：不同区域不同策略
4. **性能优化**：可用查找表（LUT）加速
5. **效果评估**：用 SSIM、PSNR 量化对比

---

*最后更新：2026-02-08*  
*适用于：视频处理、ISP Pipeline、NN 后处理*



