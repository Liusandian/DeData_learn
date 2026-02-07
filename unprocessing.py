"""
Unprocessing Pipeline: 从 sRGB 图像生成 Raw Bayer 图像
基于论文：Unprocessing Images for Learned Raw Denoising (CVPR 2019)

作者：Tim Brooks, Ben Mildenhall (Google Research)
实现：完整的逆 ISP 流程

用法：
    python unprocessing.py --input image.png --output raw_bayer.png --iso 1600
"""

import numpy as np
import cv2
import argparse
from pathlib import Path


class UnprocessingPipeline:
    """
    完整的 Unprocessing 流程
    将 sRGB 图像转换为 Raw Bayer 图像
    """
    
    def __init__(self, 
                 random_ccm=True,
                 random_gains=True,
                 add_noise=True,
                 bayer_pattern='RGGB'):
        """
        初始化 Unprocessing Pipeline
        
        Args:
            random_ccm (bool): 是否随机采样 CCM（增加数据多样性）
            random_gains (bool): 是否随机采样白平衡增益
            add_noise (bool): 是否添加噪声
            bayer_pattern (str): Bayer 模式 ('RGGB', 'BGGR', 'GRBG', 'GBRG')
        """
        self.random_ccm = random_ccm
        self.random_gains = random_gains
        self.add_noise_flag = add_noise
        self.bayer_pattern = bayer_pattern
        
        # 平均逆 CCM（来自论文，统计多个相机的平均值）
        self.ccm_inv_mean = np.array([
            [ 1.0234, -0.2969, -0.2266],
            [-0.5625,  1.6328, -0.0469],
            [-0.0703, -0.2188,  1.2891]
        ], dtype=np.float32)
        
        print("=" * 60)
        print("Unprocessing Pipeline 已初始化")
        print("=" * 60)
        print(f"随机 CCM: {random_ccm}")
        print(f"随机白平衡: {random_gains}")
        print(f"添加噪声: {add_noise}")
        print(f"Bayer 模式: {bayer_pattern}")
        print("=" * 60)
    
    def unprocess(self, srgb_image, iso=None, verbose=True):
        """
        完整的 Unprocessing 流程
        
        Args:
            srgb_image (np.ndarray): sRGB 图像, shape (H, W, 3), range [0, 255]
            iso (int): ISO 值，如果为 None 则随机选择 [400, 800, 1600, 3200]
            verbose (bool): 是否打印详细信息
        
        Returns:
            raw_bayer (np.ndarray): Raw Bayer 图像, shape (H, W), range [0, 1]
            metadata (dict): 处理过程的元数据
        """
        if verbose:
            print("\n" + "=" * 60)
            print("开始 Unprocessing 流程")
            print("=" * 60)
        
        metadata = {}
        
        # 输入验证和预处理
        if srgb_image.dtype != np.uint8:
            print(f"警告：输入图像类型为 {srgb_image.dtype}，转换为 uint8")
            srgb_image = srgb_image.astype(np.uint8)
        
        # 归一化到 [0, 1]
        img = srgb_image.astype(np.float32) / 255.0
        metadata['input_shape'] = srgb_image.shape
        metadata['input_range'] = [srgb_image.min(), srgb_image.max()]
        
        # ========== 步骤 1: 逆 Gamma 校正 ==========
        if verbose:
            print("\n步骤 1: 逆 Gamma 校正")
        linear = self.step1_inverse_gamma(img)
        if verbose:
            print(f"  输出范围: [{linear.min():.4f}, {linear.max():.4f}]")
        
        # ========== 步骤 2: 逆色调映射 ==========
        if verbose:
            print("\n步骤 2: 逆色调映射")
        linear_scaled, scale = self.step2_inverse_tone_mapping(linear)
        metadata['tone_scale'] = float(scale)
        if verbose:
            print(f"  缩放因子: {scale:.4f}")
            print(f"  输出范围: [{linear_scaled.min():.4f}, {linear_scaled.max():.4f}]")
        
        # ========== 步骤 3: 逆色彩校正 ==========
        if verbose:
            print("\n步骤 3: 逆色彩校正")
        if self.random_ccm:
            ccm_inv = self._sample_random_ccm()
            if verbose:
                print("  使用随机 CCM")
        else:
            ccm_inv = self.ccm_inv_mean
            if verbose:
                print("  使用平均 CCM")
        
        camera_rgb = self.step3_inverse_color_correction(linear_scaled, ccm_inv)
        metadata['ccm_inv'] = ccm_inv.tolist()
        if verbose:
            print(f"  输出范围: [{camera_rgb.min():.4f}, {camera_rgb.max():.4f}]")
        
        # ========== 步骤 4: 逆白平衡 ==========
        if verbose:
            print("\n步骤 4: 逆白平衡")
        camera_rgb_inv_wb, gains = self.step4_inverse_white_balance(camera_rgb)
        metadata['wb_gains'] = gains
        if verbose:
            print(f"  Red Gain:   {gains['red']:.3f}")
            print(f"  Green Gain: {gains['green']:.3f}")
            print(f"  Blue Gain:  {gains['blue']:.3f}")
            print(f"  输出范围: [{camera_rgb_inv_wb.min():.4f}, {camera_rgb_inv_wb.max():.4f}]")
        
        # ========== 步骤 5: Mosaic ==========
        if verbose:
            print("\n步骤 5: Mosaic（马赛克化）")
        bayer = self.step5_mosaic(camera_rgb_inv_wb)
        metadata['bayer_pattern'] = self.bayer_pattern
        if verbose:
            print(f"  Bayer 模式: {self.bayer_pattern}")
            print(f"  输出形状: {bayer.shape}")
            print(f"  输出范围: [{bayer.min():.4f}, {bayer.max():.4f}]")
        
        # ========== 步骤 6: 添加噪声 ==========
        if self.add_noise_flag:
            if verbose:
                print("\n步骤 6: 添加噪声")
            
            # 随机选择 ISO（如果未指定）
            if iso is None:
                iso = int(np.random.choice([400, 800, 1600, 3200]))
            
            bayer_noisy = self.step6_add_noise(bayer, iso)
            metadata['iso'] = int(iso)
            metadata['noise_added'] = True
            
            if verbose:
                print(f"  ISO: {iso}")
                noise_level = np.std(bayer_noisy - bayer)
                print(f"  噪声水平 (std): {noise_level:.6f}")
                print(f"  输出范围: [{bayer_noisy.min():.4f}, {bayer_noisy.max():.4f}]")
        else:
            bayer_noisy = bayer
            metadata['iso'] = 100
            metadata['noise_added'] = False
            if verbose:
                print("\n步骤 6: 跳过添加噪声")
        
        metadata['output_shape'] = bayer_noisy.shape
        
        if verbose:
            print("\n" + "=" * 60)
            print("Unprocessing 完成")
            print("=" * 60)
            print(f"输入: {metadata['input_shape']} (sRGB)")
            print(f"输出: {metadata['output_shape']} (Raw Bayer)")
            print("=" * 60)
        
        return bayer_noisy, metadata
    
    # ================================================================
    # 步骤 1: 逆 Gamma 校正
    # ================================================================
    def step1_inverse_gamma(self, srgb):
        """
        步骤 1: 逆 Gamma 校正
        
        目的：从 sRGB 非线性空间转回线性空间
        
        sRGB 标准公式：
            if sRGB ≤ 0.04045:
                linear = sRGB / 12.92
            else:
                linear = ((sRGB + 0.055) / 1.055)^2.4
        
        Args:
            srgb (np.ndarray): sRGB 图像, range [0, 1]
        
        Returns:
            linear (np.ndarray): 线性 RGB 图像, range [0, 1]
        """
        linear = np.where(
            srgb <= 0.04045,
            srgb / 12.92,
            np.power((srgb + 0.055) / 1.055, 2.4)
        )
        
        return linear.astype(np.float32)
    
    # ================================================================
    # 步骤 2: 逆色调映射
    # ================================================================
    def step2_inverse_tone_mapping(self, linear_rgb, percentile=99, safe_range=0.95):
        """
        步骤 2: 逆色调映射
        
        目的：扩展动态范围（近似 Raw 的高动态范围）
        
        问题：
            - sRGB: 8-bit, 动态范围 [0, 1]
            - Raw: 12-14 bit, 动态范围更大
        
        近似方法：
            1. 找到 sRGB 中的亮点（99分位）
            2. 假设这个值对应 Raw 的某个安全值（如 0.95）
            3. 线性缩放整个图像
        
        Args:
            linear_rgb (np.ndarray): 线性 RGB, range [0, 1]
            percentile (float): 分位点（避免被极值影响）
            safe_range (float): Raw 的安全上界（避免饱和）
        
        Returns:
            linear_scaled (np.ndarray): 缩放后的线性 RGB
            scale (float): 缩放因子
        """
        # 找到亮点
        max_val = np.percentile(linear_rgb, percentile)
        
        # 计算缩放因子
        scale = safe_range / (max_val + 1e-8)
        
        # 缩放
        linear_scaled = linear_rgb * scale
        
        # 裁剪
        linear_scaled = np.clip(linear_scaled, 0, 1)
        
        return linear_scaled.astype(np.float32), scale
    
    # ================================================================
    # 步骤 3: 逆色彩校正
    # ================================================================
    def step3_inverse_color_correction(self, rgb, ccm_inv):
        """
        步骤 3: 逆色彩校正
        
        目的：从 sRGB 色彩空间转到相机原生色彩空间
        
        正向 ISP：
            sRGB = CCM × Camera_RGB
        
        逆向：
            Camera_RGB = CCM^(-1) × sRGB
        
        Args:
            rgb (np.ndarray): RGB 图像, shape (H, W, 3)
            ccm_inv (np.ndarray): 逆色彩校正矩阵, shape (3, 3)
        
        Returns:
            camera_rgb (np.ndarray): 相机色彩空间的 RGB
        """
        h, w, c = rgb.shape
        
        # 重塑为 (N, 3)
        rgb_flat = rgb.reshape(-1, 3)
        
        # 应用矩阵变换: camera_rgb = rgb @ ccm_inv^T
        camera_rgb_flat = rgb_flat @ ccm_inv.T
        
        # 重塑回原形状
        camera_rgb = camera_rgb_flat.reshape(h, w, c)
        
        # 裁剪到有效范围
        camera_rgb = np.clip(camera_rgb, 0, 1)
        
        return camera_rgb.astype(np.float32)
    
    # ================================================================
    # 步骤 4: 逆白平衡
    # ================================================================
    def step4_inverse_white_balance(self, rgb):
        """
        步骤 4: 逆白平衡
        
        目的：移除白平衡校正，恢复原始色温
        
        正向 ISP：
            RGB_balanced = [R × gain_r, G × gain_g, B × gain_b]
        
        逆向：
            RGB_original = [R / gain_r, G / gain_g, B / gain_b]
        
        白平衡增益的典型范围（基于论文统计）：
            Red Gain:   1.9 - 2.4
            Green Gain: 1.0 (固定，作为基准)
            Blue Gain:  1.5 - 1.9
        
        Args:
            rgb (np.ndarray): RGB 图像, shape (H, W, 3)
        
        Returns:
            rgb_inv_wb (np.ndarray): 逆白平衡后的 RGB
            gains (dict): 使用的增益值
        """
        if self.random_gains:
            # 从相机增益的典型分布采样
            red_gain = float(np.random.uniform(1.9, 2.4))
            green_gain = 1.0
            blue_gain = float(np.random.uniform(1.5, 1.9))
        else:
            # 使用平均值
            red_gain = 2.15
            green_gain = 1.0
            blue_gain = 1.7
        
        # 应用逆增益
        rgb_inv_wb = rgb.copy()
        rgb_inv_wb[:, :, 0] /= red_gain    # R 通道
        rgb_inv_wb[:, :, 1] /= green_gain  # G 通道
        rgb_inv_wb[:, :, 2] /= blue_gain   # B 通道
        
        # 裁剪到有效范围
        rgb_inv_wb = np.clip(rgb_inv_wb, 0, 1)
        
        gains = {
            'red': red_gain,
            'green': green_gain,
            'blue': blue_gain
        }
        
        return rgb_inv_wb.astype(np.float32), gains
    
    # ================================================================
    # 步骤 5: Mosaic（马赛克化）
    # ================================================================
    def step5_mosaic(self, rgb):
        """
        步骤 5: Mosaic（马赛克化）
        
        目的：从 RGB 三通道转回 Bayer 单通道
        
        Bayer Pattern (RGGB):
            R  G  R  G  R  G
            G  B  G  B  G  B
            R  G  R  G  R  G
            G  B  G  B  G  B
        
        每个像素只保留一个颜色分量
        
        Args:
            rgb (np.ndarray): RGB 图像, shape (H, W, 3)
        
        Returns:
            bayer (np.ndarray): Bayer 图像, shape (H, W)
        """
        h, w, c = rgb.shape
        bayer = np.zeros((h, w), dtype=np.float32)
        
        if self.bayer_pattern == 'RGGB':
            # R  G
            # G  B
            bayer[0::2, 0::2] = rgb[0::2, 0::2, 0]  # R 位置
            bayer[0::2, 1::2] = rgb[0::2, 1::2, 1]  # G 位置（R 行）
            bayer[1::2, 0::2] = rgb[1::2, 0::2, 1]  # G 位置（B 行）
            bayer[1::2, 1::2] = rgb[1::2, 1::2, 2]  # B 位置
            
        elif self.bayer_pattern == 'BGGR':
            # B  G
            # G  R
            bayer[0::2, 0::2] = rgb[0::2, 0::2, 2]  # B
            bayer[0::2, 1::2] = rgb[0::2, 1::2, 1]  # G
            bayer[1::2, 0::2] = rgb[1::2, 0::2, 1]  # G
            bayer[1::2, 1::2] = rgb[1::2, 1::2, 0]  # R
            
        elif self.bayer_pattern == 'GRBG':
            # G  R
            # B  G
            bayer[0::2, 0::2] = rgb[0::2, 0::2, 1]  # G
            bayer[0::2, 1::2] = rgb[0::2, 1::2, 0]  # R
            bayer[1::2, 0::2] = rgb[1::2, 0::2, 2]  # B
            bayer[1::2, 1::2] = rgb[1::2, 1::2, 1]  # G
            
        elif self.bayer_pattern == 'GBRG':
            # G  B
            # R  G
            bayer[0::2, 0::2] = rgb[0::2, 0::2, 1]  # G
            bayer[0::2, 1::2] = rgb[0::2, 1::2, 2]  # B
            bayer[1::2, 0::2] = rgb[1::2, 0::2, 0]  # R
            bayer[1::2, 1::2] = rgb[1::2, 1::2, 1]  # G
        
        else:
            raise ValueError(f"不支持的 Bayer 模式: {self.bayer_pattern}")
        
        return bayer
    
    # ================================================================
    # 步骤 6: 添加噪声
    # ================================================================
    def step6_add_noise(self, bayer, iso):
        """
        步骤 6: 添加真实的相机噪声
        
        目的：模拟相机传感器的噪声
        
        噪声模型（泊松-高斯组合）：
            y = Poisson(x × gain) / gain + Gaussian(0, σ_read²)
        
        物理解释：
            1. 光子噪声（Poisson）- 信号相关
               - 来源：光子到达的量子随机性
               - 特性：方差 = 均值
               - 亮部噪声大，暗部噪声小
            
            2. 读取噪声（Gaussian）- 信号无关
               - 来源：电路噪声
               - 特性：固定方差
               - 与信号强度无关
        
        Args:
            bayer (np.ndarray): Bayer 图像, range [0, 1]
            iso (int): ISO 值 (100, 200, 400, ..., 6400)
        
        Returns:
            bayer_noisy (np.ndarray): 带噪声的 Bayer 图像
        """
        # ISO 增益
        gain = iso / 100.0
        
        # 光子噪声参数
        shot_noise_scale = gain
        
        # 读取噪声参数（论文的经验公式）
        read_noise_std = 0.0005 * np.sqrt(gain)
        
        # ===== 添加泊松噪声（光子噪声）=====
        # 缩放到光子计数范围
        bayer_scaled = bayer * shot_noise_scale
        
        # 泊松采样（必须是非负值）
        bayer_noisy = np.random.poisson(
            np.clip(bayer_scaled, 0, None)
        ).astype(np.float32) / shot_noise_scale
        
        # ===== 添加高斯噪声（读取噪声）=====
        read_noise = np.random.normal(0, read_noise_std, bayer.shape)
        bayer_noisy = bayer_noisy + read_noise
        
        # 裁剪到有效范围
        bayer_noisy = np.clip(bayer_noisy, 0, 1)
        
        return bayer_noisy.astype(np.float32)
    
    # ================================================================
    # 辅助函数
    # ================================================================
    def _sample_random_ccm(self):
        """
        采样随机 CCM（增加数据多样性）
        
        在平均 CCM 附近添加随机扰动
        """
        noise = np.random.normal(0, 0.05, (3, 3))
        ccm_inv_random = self.ccm_inv_mean + noise
        return ccm_inv_random.astype(np.float32)


# ====================================================================
# 辅助函数：保存和加载
# ====================================================================
def save_raw_bayer(bayer, filename, bit_depth=12):
    """
    保存 Raw Bayer 图像
    
    Args:
        bayer (np.ndarray): Bayer 图像, range [0, 1]
        filename (str): 保存路径
        bit_depth (int): 位深度（12 或 14）
    """
    # 转换到指定位深度
    max_val = 2 ** bit_depth - 1
    bayer_int = (bayer * max_val).astype(np.uint16)
    
    # 保存为 16-bit PNG
    cv2.imwrite(filename, bayer_int)
    
    print(f"\n已保存 Raw Bayer 图像到: {filename}")
    print(f"  形状: {bayer_int.shape}")
    print(f"  数据类型: {bayer_int.dtype}")
    print(f"  值范围: [{bayer_int.min()}, {bayer_int.max()}]")
    print(f"  位深度: {bit_depth}-bit")


def load_raw_bayer(filename, bit_depth=12):
    """
    加载 Raw Bayer 图像
    
    Args:
        filename (str): 文件路径
        bit_depth (int): 位深度
    
    Returns:
        bayer (np.ndarray): Bayer 图像, range [0, 1]
    """
    # 读取为 16-bit
    bayer_int = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    
    # 归一化
    max_val = 2 ** bit_depth - 1
    bayer = bayer_int.astype(np.float32) / max_val
    
    return bayer


def visualize_bayer(bayer, save_path=None):
    """
    可视化 Bayer 图像（简单去马赛克后）
    
    Args:
        bayer (np.ndarray): Bayer 图像, range [0, 1]
        save_path (str): 保存路径
    """
    # 转为 uint8
    bayer_uint8 = (bayer * 255).astype(np.uint8)
    
    # 简单去马赛克
    rgb = cv2.cvtColor(bayer_uint8, cv2.COLOR_BAYER_RGGB2RGB)
    
    # 显示
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(bayer, cmap='gray')
    axes[0].set_title('Raw Bayer (单通道)')
    axes[0].axis('off')
    
    axes[1].imshow(rgb)
    axes[1].set_title('简单去马赛克后')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"可视化已保存到: {save_path}")
    else:
        plt.show()


# ====================================================================
# 主函数
# ====================================================================
def main():
    """主函数：命令行接口"""
    
    parser = argparse.ArgumentParser(
        description='Unprocessing: 从 sRGB 图像生成 Raw Bayer 图像'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='输入 sRGB 图像路径'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='output_raw.png',
        help='输出 Raw Bayer 图像路径'
    )
    parser.add_argument(
        '--iso',
        type=int,
        default=None,
        choices=[100, 200, 400, 800, 1600, 3200, 6400],
        help='ISO 值（默认随机选择）'
    )
    parser.add_argument(
        '--no-noise',
        action='store_true',
        help='不添加噪声'
    )
    parser.add_argument(
        '--bayer-pattern',
        type=str,
        default='RGGB',
        choices=['RGGB', 'BGGR', 'GRBG', 'GBRG'],
        help='Bayer 模式'
    )
    parser.add_argument(
        '--bit-depth',
        type=int,
        default=12,
        choices=[12, 14],
        help='Raw 图像位深度'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='可视化结果'
    )
    
    args = parser.parse_args()
    
    # 读取输入图像
    print(f"\n读取输入图像: {args.input}")
    srgb_image = cv2.imread(args.input)
    
    if srgb_image is None:
        print(f"错误：无法读取图像 {args.input}")
        return
    
    # OpenCV 读取的是 BGR，转为 RGB
    srgb_image = cv2.cvtColor(srgb_image, cv2.COLOR_BGR2RGB)
    print(f"  图像形状: {srgb_image.shape}")
    print(f"  图像类型: {srgb_image.dtype}")
    
    # 创建 Unprocessing Pipeline
    unprocessor = UnprocessingPipeline(
        random_ccm=True,
        random_gains=True,
        add_noise=not args.no_noise,
        bayer_pattern=args.bayer_pattern
    )
    
    # Unprocessing
    raw_bayer, metadata = unprocessor.unprocess(
        srgb_image,
        iso=args.iso,
        verbose=True
    )
    
    # 保存 Raw Bayer 图像
    save_raw_bayer(raw_bayer, args.output, bit_depth=args.bit_depth)
    
    # 保存元数据
    metadata_path = args.output.replace('.png', '_metadata.txt')
    with open(metadata_path, 'w') as f:
        f.write("Unprocessing Metadata\n")
        f.write("=" * 60 + "\n")
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    
    print(f"\n元数据已保存到: {metadata_path}")
    
    # 可视化
    if args.visualize:
        vis_path = args.output.replace('.png', '_visualization.png')
        visualize_bayer(raw_bayer, save_path=vis_path)
    
    print("\n" + "=" * 60)
    print("处理完成！")
    print("=" * 60)


# ====================================================================
# 批处理函数
# ====================================================================
def batch_process(input_dir, output_dir, num_samples=100, iso_range=None):
    """
    批量处理：生成大量训练数据
    
    Args:
        input_dir (str): 输入 sRGB 图像目录
        output_dir (str): 输出目录
        num_samples (int): 生成样本数
        iso_range (list): ISO 范围，如 [400, 800, 1600, 3200]
    """
    from glob import glob
    from tqdm import tqdm
    import os
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/raw_noisy", exist_ok=True)
    os.makedirs(f"{output_dir}/raw_clean", exist_ok=True)
    
    # 获取输入图像
    image_paths = glob(f"{input_dir}/*.png") + glob(f"{input_dir}/*.jpg")
    print(f"找到 {len(image_paths)} 张输入图像")
    
    if len(image_paths) == 0:
        print(f"错误：在 {input_dir} 中未找到图像")
        return
    
    # ISO 范围
    if iso_range is None:
        iso_range = [400, 800, 1600, 3200]
    
    # 创建 Unprocessing Pipeline
    unprocessor = UnprocessingPipeline(
        random_ccm=True,
        random_gains=True,
        add_noise=False  # 先不加噪声，生成干净 Raw
    )
    
    # 批量处理
    print(f"\n开始批量处理，生成 {num_samples} 对样本...")
    
    for i in tqdm(range(num_samples), desc="生成 Raw 数据"):
        # 随机选择一张图像
        img_path = np.random.choice(image_paths)
        srgb = cv2.imread(img_path)
        srgb = cv2.cvtColor(srgb, cv2.COLOR_BGR2RGB)
        
        # 随机裁剪 patch（如果图像够大）
        h, w = srgb.shape[:2]
        patch_size = 512
        
        if h >= patch_size and w >= patch_size:
            top = np.random.randint(0, h - patch_size + 1)
            left = np.random.randint(0, w - patch_size + 1)
            srgb_patch = srgb[top:top+patch_size, left:left+patch_size]
        else:
            srgb_patch = srgb
        
        # Unprocessing（生成干净 Raw）
        clean_raw, metadata = unprocessor.unprocess(srgb_patch, verbose=False)
        
        # 添加噪声（生成噪声 Raw）
        iso = np.random.choice(iso_range)
        noisy_raw = unprocessor.step6_add_noise(clean_raw, iso)
        
        # 保存
        clean_filename = f"{output_dir}/raw_clean/{i:05d}.png"
        noisy_filename = f"{output_dir}/raw_noisy/{i:05d}_iso{iso}.png"
        
        save_raw_bayer(clean_raw, clean_filename, bit_depth=12)
        save_raw_bayer(noisy_raw, noisy_filename, bit_depth=12)
    
    print(f"\n批量处理完成！")
    print(f"输出目录: {output_dir}")
    print(f"  干净 Raw: {output_dir}/raw_clean/")
    print(f"  噪声 Raw: {output_dir}/raw_noisy/")


# ====================================================================
# 运行
# ====================================================================
if __name__ == '__main__':
    # 如果有命令行参数，运行 main
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        # 否则运行示例
        print("=" * 60)
        print("Unprocessing Pipeline 示例")
        print("=" * 60)
        print("\n用法示例：")
        print("  python unprocessing.py --input image.png --output raw.png --iso 1600")
        print("  python unprocessing.py --input image.png --output raw.png --no-noise")
        print("  python unprocessing.py --input image.png --visualize")
        print("\n批量处理示例（在代码中调用）：")
        print("  from unprocessing import batch_process")
        print("  batch_process('srgb_dir', 'output_dir', num_samples=1000)")
        print("\n" + "=" * 60)

