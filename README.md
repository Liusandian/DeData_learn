# DeData_learn

存放数据退化（dedata,逆ISP)相关的学习脚本，来源包括不限于cursor,自开发脚本等等

## 📁 项目结构

```
DeData_learn/
├── unprocessing.py              # 逆ISP Pipeline实现（含可视化）
├── test_unprocessing.py         # 单元测试
├── test_visualization.py        # 可视化功能测试脚本 ⭐新增
├── docus/                       # 文档目录
│   ├── AI-ISP数据退化详解.md
│   ├── 从HighLevel到LowLevel-AI-ISP学习路线图.md
│   ├── NAFNet1x-W16A8量化伪影分析-紫边与鬼影.md
│   └── Unprocessing可视化使用指南.md  ⭐新增
└── README.md
```

## 🚀 快速开始

### 1. 基本使用（带可视化）

```bash
# 处理图像并生成每个步骤的可视化
python unprocessing.py \
    --input your_image.jpg \
    --output raw_bayer.png \
    --iso 1600 \
    --visualize \
    --vis-dir visualization
```

### 2. 快速测试可视化功能

```bash
# 运行测试脚本，自动生成所有可视化
python test_visualization.py
```

## 🎯 核心功能

### Unprocessing Pipeline（逆ISP流程）

将 sRGB 图像转换为 Raw Bayer 图像，模拟相机传感器的原始数据

**6个关键步骤：**

1. **逆 Gamma 校正** - sRGB → 线性 RGB
2. **逆色调映射** - 扩展动态范围（LDR → HDR）
3. **逆色彩校正** - sRGB 色彩空间 → 相机原生色彩空间
4. **逆白平衡** - 移除色温校正，恢复原始色温
5. **Mosaic 马赛克化** - RGB 三通道 → Bayer 单通道
6. **添加噪声** - 模拟传感器噪声（泊松-高斯组合）

### ⭐ 新增：可视化功能

每个步骤都会生成**左右对比图**，方便理解数据退化过程：

- `step1_inverse_gamma.png` - 逆Gamma校正
- `step2_inverse_tone_mapping.png` - 逆色调映射
- `step3_inverse_color_correction.png` - 逆色彩校正
- `step4_inverse_white_balance.png` - 逆白平衡
- `step5_mosaic.png` - Mosaic马赛克化
- `step6_add_noise.png` - 添加噪声
- `step6_noise_detail.png` - 噪声详细分析
- `pipeline_summary.png` - **完整流程总结** ⭐必看

## 📚 文档

### 详细文档
- **[Unprocessing可视化使用指南](docus/Unprocessing可视化使用指南.md)** - 详细的可视化功能说明
- **[AI-ISP数据退化详解](docus/AI-ISP数据退化详解.md)** - ISP原理和数据退化
- **[NAFNet量化伪影分析](docus/NAFNet1x-W16A8量化伪影分析-紫边与鬼影.md)** - 量化问题分析

### 学习路线
- **[从HighLevel到LowLevel-AI-ISP学习路线图](docus/从HighLevel到LowLevel-AI-ISP学习路线图.md)**

## 💡 使用示例

### Python代码中调用

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
    output_dir='visualization' # 可视化保存目录
)

# 处理（会自动保存所有可视化）
raw_bayer, metadata = unprocessor.unprocess(img, iso=1600, verbose=True)

# 查看 visualization/ 目录下的所有可视化结果
```

### 批量处理

```python
from unprocessing import batch_process

# 批量生成训练数据
batch_process(
    input_dir='srgb_images',
    output_dir='raw_dataset',
    num_samples=1000,
    iso_range=[400, 800, 1600, 3200]
)
```

## 🔧 依赖

```bash
pip install numpy opencv-python matplotlib
```

## 📖 参考

- **论文**: Unprocessing Images for Learned Raw Denoising (CVPR 2019)
- **作者**: Tim Brooks, Ben Mildenhall (Google Research)

## 🎓 适用场景

- 低级视觉任务（Low-level Vision）
- Raw图像去噪
- ISP学习和研究
- 数据增强（Data Augmentation）
- 图像退化模拟

---

*最后更新：2026-02-08*
