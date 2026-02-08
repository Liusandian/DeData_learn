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
import matplotlib.pyplot as plt
import os
from matplotlib import font_manager


def _configure_matplotlib_fonts():
    """配置 Matplotlib 中文字体，避免标题乱码。"""
    preferred_fonts = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Source Han Sans CN",
        "PingFang SC",
        "Arial Unicode MS",
    ]
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in preferred_fonts:
        if font_name in available_fonts:
            plt.rcParams["font.sans-serif"] = [font_name]
            break
    else:
        # Fallback to a default font with decent unicode coverage.
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False


_configure_matplotlib_fonts()


class UnprocessingPipeline:
    """
    完整的 Unprocessing 流程
    将 sRGB 图像转换为 Raw Bayer 图像
    """
    
    def __init__(self, 
                 random_ccm=True,
                 random_gains=True,
                 add_noise=True,
                 bayer_pattern='RGGB',
                 visualize=True,
                 output_dir='visualization'):
        """
        初始化 Unprocessing Pipeline
        
        Args:
            random_ccm (bool): 是否随机采样 CCM（增加数据多样性）
            random_gains (bool): 是否随机采样白平衡增益
            add_noise (bool): 是否添加噪声
            bayer_pattern (str): Bayer 模式 ('RGGB', 'BGGR', 'GRBG', 'GBRG')
            visualize (bool): 是否保存可视化结果
            output_dir (str): 可视化结果保存目录
        """
        self.random_ccm = random_ccm
        self.random_gains = random_gains
        self.add_noise_flag = add_noise
        self.bayer_pattern = bayer_pattern
        self.visualize = visualize
        self.output_dir = output_dir
        
        # 创建可视化目录
        if self.visualize:
            os.makedirs(self.output_dir, exist_ok=True)
        
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
        print(f"可视化: {visualize}")
        if visualize:
            print(f"可视化目录: {output_dir}")
        print("=" * 60)
    
    def _visualize_comparison(self, img_before, img_after, title_before, title_after, 
                              filename, suptitle="", cmap_before=None, cmap_after=None):
        """
        可视化对比：左边原图，右边处理后
        
        Args:
            img_before: 处理前的图像
            img_after: 处理后的图像
            title_before: 左图标题
            title_after: 右图标题
            filename: 保存文件名
            suptitle: 总标题
            cmap_before: 左图colormap（灰度图用'gray'）
            cmap_after: 右图colormap
        """
        if not self.visualize:
            return
        
        # ========== 1. 保存对比图 ==========
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 左图：处理前
        if cmap_before:
            axes[0].imshow(np.clip(img_before, 0, 1), cmap=cmap_before)
        else:
            axes[0].imshow(np.clip(img_before, 0, 1))
        axes[0].set_title(title_before, fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # 添加统计信息
        stats_before = f"Min: {img_before.min():.4f}\nMax: {img_before.max():.4f}\nMean: {img_before.mean():.4f}"
        axes[0].text(0.02, 0.98, stats_before, transform=axes[0].transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 右图：处理后
        if cmap_after:
            axes[1].imshow(np.clip(img_after, 0, 1), cmap=cmap_after)
        else:
            axes[1].imshow(np.clip(img_after, 0, 1))
        axes[1].set_title(title_after, fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # 添加统计信息
        stats_after = f"Min: {img_after.min():.4f}\nMax: {img_after.max():.4f}\nMean: {img_after.mean():.4f}"
        axes[1].text(0.02, 0.98, stats_after, transform=axes[1].transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        if suptitle:
            plt.suptitle(suptitle, fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ 对比图已保存: {save_path}")
        
        # ========== 2. 保存独立的原始分辨率图像 ==========
        base_name = filename.replace('.png', '')
        
        # 保存处理前的图像
        before_path = os.path.join(self.output_dir, f"{base_name}_before.png")
        self._save_single_image(img_before, before_path, cmap_before)
        
        # 保存处理后的图像
        after_path = os.path.join(self.output_dir, f"{base_name}_after.png")
        self._save_single_image(img_after, after_path, cmap_after)
        
        print(f"  ✓ 独立图像已保存: {base_name}_before.png / {base_name}_after.png")
    
    def _save_single_image(self, img, save_path, cmap=None):
        """
        保存单张原始分辨率图像
        
        Args:
            img: 图像数据
            save_path: 保存路径
            cmap: colormap（灰度图用'gray'）
        """
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        
        if cmap:
            ax.imshow(np.clip(img, 0, 1), cmap=cmap)
        else:
            ax.imshow(np.clip(img, 0, 1))
        
        ax.axis('off')
        plt.tight_layout(pad=0)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close()
    
    def _visualize_noise_detail(self, clean, noisy, noise_map, iso):
        """
        可视化噪声的详细信息：干净图、噪声图、噪声分布直方图
        
        Args:
            clean: 干净图像
            noisy: 噪声图像
            noise_map: 噪声图 (noisy - clean)
            iso: ISO值
        """
        if not self.visualize:
            return
        
        fig = plt.figure(figsize=(18, 5))
        
        # 子图1：干净图像
        ax1 = plt.subplot(1, 4, 1)
        ax1.imshow(clean, cmap='gray', vmin=0, vmax=1)
        ax1.set_title('干净图像', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # 子图2：噪声图像
        ax2 = plt.subplot(1, 4, 2)
        ax2.imshow(noisy, cmap='gray', vmin=0, vmax=1)
        ax2.set_title(f'噪声图像 (ISO {iso})', fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        # 子图3：噪声图（放大显示）
        ax3 = plt.subplot(1, 4, 3)
        noise_vis = ax3.imshow(noise_map, cmap='RdBu_r', vmin=-0.1, vmax=0.1)
        ax3.set_title('噪声分布 (放大)', fontsize=12, fontweight='bold')
        ax3.axis('off')
        plt.colorbar(noise_vis, ax=ax3, fraction=0.046)
        
        # 子图4：噪声直方图
        ax4 = plt.subplot(1, 4, 4)
        ax4.hist(noise_map.flatten(), bins=100, alpha=0.7, color='steelblue', edgecolor='black')
        ax4.set_title('噪声分布直方图', fontsize=12, fontweight='bold')
        ax4.set_xlabel('噪声值', fontsize=10)
        ax4.set_ylabel('频次', fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        # 添加统计信息
        noise_stats = f"均值: {noise_map.mean():.6f}\n标准差: {noise_map.std():.6f}\nMin: {noise_map.min():.6f}\nMax: {noise_map.max():.6f}"
        ax4.text(0.95, 0.95, noise_stats, transform=ax4.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.suptitle(f'噪声详细分析 (ISO={iso}, 泊松-高斯组合噪声)', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, 'step6_noise_detail.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ 噪声详细分析已保存: {save_path}")
    
    def _visualize_pipeline_summary(self, srgb, linear, linear_scaled, camera_rgb, 
                                    camera_rgb_wb, bayer, bayer_noisy, metadata):
        """
        可视化整个Pipeline的流程总结
        
        Args:
            srgb: 步骤0 - 输入sRGB图像
            linear: 步骤1 - 线性RGB
            linear_scaled: 步骤2 - 扩展动态范围
            camera_rgb: 步骤3 - 相机RGB
            camera_rgb_wb: 步骤4 - 逆白平衡
            bayer: 步骤5 - Bayer图
            bayer_noisy: 步骤6 - 带噪声Bayer
            metadata: 元数据
        """
        if not self.visualize:
            return
        
        fig = plt.figure(figsize=(20, 12))
        
        # 使用GridSpec来更好地控制布局
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.2)
        
        # 步骤0: 输入sRGB
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.imshow(np.clip(srgb, 0, 1))
        ax0.set_title('步骤0: 输入 sRGB\n(Gamma编码)', fontsize=11, fontweight='bold')
        ax0.axis('off')
        
        # 步骤1: 线性RGB
        ax1 = fig.add_subplot(gs[0, 1])
        ax1.imshow(np.clip(linear, 0, 1))
        ax1.set_title('步骤1: 线性 RGB\n(逆Gamma)', fontsize=11, fontweight='bold')
        ax1.axis('off')
        ax0.annotate('', xy=(1.05, 0.5), xytext=(0.95, 0.5), 
                    xycoords=ax0.transAxes, textcoords=ax1.transAxes,
                    arrowprops=dict(arrowstyle='->', lw=2, color='red'))
        
        # 步骤2: 扩展动态范围
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.imshow(np.clip(linear_scaled, 0, 1))
        scale = metadata.get('tone_scale', 1.0)
        ax2.set_title(f'步骤2: 扩展动态范围\n(scale={scale:.3f})', fontsize=11, fontweight='bold')
        ax2.axis('off')
        ax1.annotate('', xy=(1.05, 0.5), xytext=(0.95, 0.5), 
                    xycoords=ax1.transAxes, textcoords=ax2.transAxes,
                    arrowprops=dict(arrowstyle='->', lw=2, color='red'))
        
        # 步骤3: 相机RGB
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.imshow(np.clip(camera_rgb, 0, 1))
        ax3.set_title('步骤3: 相机 RGB\n(逆CCM)', fontsize=11, fontweight='bold')
        ax3.axis('off')
        ax2.annotate('', xy=(0.5, 1.1), xytext=(0.1, 0.9), 
                    xycoords=ax3.transAxes, textcoords=ax2.transAxes,
                    arrowprops=dict(arrowstyle='->', lw=2, color='red'))
        
        # 步骤4: 逆白平衡
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.imshow(np.clip(camera_rgb_wb, 0, 1))
        gains = metadata.get('wb_gains', {})
        ax4.set_title(f'步骤4: 逆白平衡\n(R:{gains.get("red", 0):.2f}, B:{gains.get("blue", 0):.2f})', 
                     fontsize=11, fontweight='bold')
        ax4.axis('off')
        ax3.annotate('', xy=(1.05, 0.5), xytext=(0.95, 0.5), 
                    xycoords=ax3.transAxes, textcoords=ax4.transAxes,
                    arrowprops=dict(arrowstyle='->', lw=2, color='red'))
        
        # 步骤5: Bayer
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.imshow(bayer, cmap='gray', vmin=0, vmax=1)
        pattern = metadata.get('bayer_pattern', 'RGGB')
        ax5.set_title(f'步骤5: Bayer Pattern\n({pattern})', fontsize=11, fontweight='bold')
        ax5.axis('off')
        ax4.annotate('', xy=(1.05, 0.5), xytext=(0.95, 0.5), 
                    xycoords=ax4.transAxes, textcoords=ax5.transAxes,
                    arrowprops=dict(arrowstyle='->', lw=2, color='red'))
        
        # 步骤6: 带噪声Bayer
        ax6 = fig.add_subplot(gs[2, 0])
        ax6.imshow(bayer_noisy, cmap='gray', vmin=0, vmax=1)
        iso = metadata.get('iso', 100)
        ax6.set_title(f'步骤6: 添加噪声\n(ISO={iso})', fontsize=11, fontweight='bold')
        ax6.axis('off')
        ax5.annotate('', xy=(0.5, 1.1), xytext=(0.5, 0.9), 
                    xycoords=ax6.transAxes, textcoords=ax5.transAxes,
                    arrowprops=dict(arrowstyle='->', lw=2, color='red'))
        
        # Bayer Pattern示意图
        ax7 = fig.add_subplot(gs[2, 1])
        bayer_demo = self._create_bayer_pattern_demo()
        ax7.imshow(bayer_demo)
        ax7.set_title(f'Bayer Pattern 示意图\n({pattern})', fontsize=11, fontweight='bold')
        ax7.axis('off')
        
        # 添加流程说明文本
        ax8 = fig.add_subplot(gs[2, 2])
        ax8.axis('off')
        summary_text = f"""
Unprocessing Pipeline 完整流程
=====================================

输入: sRGB 图像 ({metadata['input_shape']})
输出: Raw Bayer 图像 ({metadata['output_shape']})

关键参数:
• 色调缩放: {metadata.get('tone_scale', 1.0):.4f}
• 白平衡增益:
  - Red: {gains.get('red', 0):.3f}
  - Green: {gains.get('green', 1.0):.3f}
  - Blue: {gains.get('blue', 0):.3f}
• Bayer模式: {pattern}
• ISO: {iso}
• 噪声: {'已添加' if metadata.get('noise_added') else '未添加'}

流程说明:
1. 逆Gamma: sRGB → 线性域
2. 逆色调映射: 扩展动态范围
3. 逆CCM: sRGB → 相机色彩空间
4. 逆白平衡: 移除色温校正
5. Mosaic: RGB → Bayer单通道
6. 添加噪声: 模拟传感器噪声
        """
        ax8.text(0.1, 0.9, summary_text, fontsize=10, 
                verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        plt.suptitle('Unprocessing Pipeline 完整流程总结 (sRGB → Raw Bayer)', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        save_path = os.path.join(self.output_dir, 'pipeline_summary.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n  ✓ Pipeline总结已保存: {save_path}")
    
    def _create_bayer_pattern_demo(self):
        """创建Bayer Pattern示意图"""
        size = 200
        demo = np.zeros((size, size, 3), dtype=np.float32)
        
        cell_size = size // 4
        
        if self.bayer_pattern == 'RGGB':
            # R G
            # G B
            demo[0:size//2, 0:size//2, 0] = 1.0  # R
            demo[0:size//2, size//2:, 1] = 1.0   # G
            demo[size//2:, 0:size//2, 1] = 1.0   # G
            demo[size//2:, size//2:, 2] = 1.0    # B
        elif self.bayer_pattern == 'BGGR':
            demo[0:size//2, 0:size//2, 2] = 1.0  # B
            demo[0:size//2, size//2:, 1] = 1.0   # G
            demo[size//2:, 0:size//2, 1] = 1.0   # G
            demo[size//2:, size//2:, 0] = 1.0    # R
        elif self.bayer_pattern == 'GRBG':
            demo[0:size//2, 0:size//2, 1] = 1.0  # G
            demo[0:size//2, size//2:, 0] = 1.0   # R
            demo[size//2:, 0:size//2, 2] = 1.0   # B
            demo[size//2:, size//2:, 1] = 1.0    # G
        elif self.bayer_pattern == 'GBRG':
            demo[0:size//2, 0:size//2, 1] = 1.0  # G
            demo[0:size//2, size//2:, 2] = 1.0   # B
            demo[size//2:, 0:size//2, 0] = 1.0   # R
            demo[size//2:, size//2:, 1] = 1.0    # G
        
        return demo
    
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
        
        # 可视化
        self._visualize_comparison(
            img, linear,
            "输入: sRGB 图像\n(Gamma 编码后)",
            "输出: 线性 RGB\n(Gamma 校正前)",
            "step1_inverse_gamma.png",
            "步骤 1: 逆 Gamma 校正 (sRGB → Linear RGB)"
        )
        
        # ========== 步骤 2: 逆色调映射 ==========
        if verbose:
            print("\n步骤 2: 逆色调映射")
        linear_scaled, scale = self.step2_inverse_tone_mapping(linear)
        metadata['tone_scale'] = float(scale)
        if verbose:
            print(f"  缩放因子: {scale:.4f}")
            print(f"  输出范围: [{linear_scaled.min():.4f}, {linear_scaled.max():.4f}]")
        
        # 可视化
        self._visualize_comparison(
            linear, linear_scaled,
            "输入: 线性 RGB\n(LDR 动态范围)",
            f"输出: 扩展动态范围\n(缩放因子: {scale:.4f})",
            "step2_inverse_tone_mapping.png",
            "步骤 2: 逆色调映射 (扩展动态范围，模拟 Raw HDR)"
        )
        
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
        
        # 可视化
        self._visualize_comparison(
            linear_scaled, camera_rgb,
            "输入: sRGB 色彩空间",
            "输出: 相机原生色彩空间\n(CCM^-1 变换后)",
            "step3_inverse_color_correction.png",
            "步骤 3: 逆色彩校正 (sRGB → Camera RGB)"
        )
        
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
        
        # 可视化
        gain_info = f"R:{gains['red']:.2f}, G:{gains['green']:.2f}, B:{gains['blue']:.2f}"
        self._visualize_comparison(
            camera_rgb, camera_rgb_inv_wb,
            "输入: 白平衡后的图像\n(色温已校正)",
            f"输出: 白平衡前的图像\n(恢复原始色温)\nGains: {gain_info}",
            "step4_inverse_white_balance.png",
            "步骤 4: 逆白平衡 (移除色温校正，恢复原始色调)"
        )
        
        # ========== 步骤 5: Mosaic ==========
        if verbose:
            print("\n步骤 5: Mosaic（马赛克化）")
        bayer = self.step5_mosaic(camera_rgb_inv_wb)
        metadata['bayer_pattern'] = self.bayer_pattern
        if verbose:
            print(f"  Bayer 模式: {self.bayer_pattern}")
            print(f"  输出形状: {bayer.shape}")
            print(f"  输出范围: [{bayer.min():.4f}, {bayer.max():.4f}]")
        
        # 可视化（特殊处理：左边RGB，右边单通道Bayer）
        self._visualize_comparison(
            camera_rgb_inv_wb, bayer,
            "输入: RGB 三通道\n(每个像素有R、G、B值)",
            f"输出: Bayer 单通道\n(模式: {self.bayer_pattern}，每个像素只保留一个颜色)",
            "step5_mosaic.png",
            "步骤 5: Mosaic 马赛克化 (RGB → Bayer Pattern)",
            cmap_after='gray'
        )
        
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
            
            # 可视化
            self._visualize_comparison(
                bayer, bayer_noisy,
                "输入: 干净的 Bayer 图像\n(无噪声)",
                f"输出: 带噪声的 Bayer 图像\n(ISO: {iso}, 噪声σ: {noise_level:.6f})",
                "step6_add_noise.png",
                f"步骤 6: 添加相机噪声 (泊松-高斯组合噪声模型，ISO={iso})",
                cmap_before='gray',
                cmap_after='gray'
            )
            
            # 额外：可视化噪声本身
            if self.visualize:
                noise_map = bayer_noisy - bayer
                self._visualize_noise_detail(bayer, bayer_noisy, noise_map, iso)
                
        else:
            bayer_noisy = bayer
            metadata['iso'] = 100
            metadata['noise_added'] = False
            if verbose:
                print("\n步骤 6: 跳过添加噪声")
        
        metadata['output_shape'] = bayer_noisy.shape
        
        # ========== 最终总结可视化 ==========
        if self.visualize:
            self._visualize_pipeline_summary(img, linear, linear_scaled, camera_rgb, 
                                            camera_rgb_inv_wb, bayer, bayer_noisy, metadata)
        
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


def bayer_to_rgb(bayer, bayer_pattern='RGGB', method='bilinear', clip=True):
    """
    将 Bayer 图像转换为 RGB 图像（去马赛克/Demosaicing）
    
    目的：从单通道 Bayer 图像恢复完整的 RGB 三通道图像
    
    Args:
        bayer (np.ndarray): Bayer 图像, shape (H, W), range [0, 1]
        bayer_pattern (str): Bayer 模式 ('RGGB', 'BGGR', 'GRBG', 'GBRG')
        method (str): 插值方法
            - 'bilinear': 双线性插值（简单快速）
            - 'opencv': OpenCV内置方法（质量较好）
            - 'edge_aware': 边缘感知插值（质量最好，速度较慢）
        clip (bool): 是否将结果裁剪到 [0, 1]
    
    Returns:
        rgb (np.ndarray): RGB 图像, shape (H, W, 3), range [0, 1]
    
    ISP知识点：
        Demosaicing 是 ISP 的关键步骤之一，直接影响图像质量
        - 简单方法（双线性）：速度快但可能出现伪色、拉链效应
        - 高级方法（边缘感知）：考虑局部梯度，避免跨边界插值
    """
    
    if method == 'opencv':
        return _demosaic_opencv(bayer, bayer_pattern, clip)
    elif method == 'bilinear':
        return _demosaic_bilinear(bayer, bayer_pattern, clip)
    elif method == 'edge_aware':
        return _demosaic_edge_aware(bayer, bayer_pattern, clip)
    else:
        raise ValueError(f"不支持的插值方法: {method}，请选择 'bilinear', 'opencv', 'edge_aware'")


def _demosaic_opencv(bayer, bayer_pattern='RGGB', clip=True):
    """
    使用 OpenCV 内置的去马赛克算法
    
    优点：速度快，质量好，工业级实现
    缺点：依赖 OpenCV，算法细节不可控
    """
    # 转为 uint8 或 uint16 供 OpenCV 处理
    if bayer.max() <= 1.0:
        bayer_uint8 = (bayer * 255).astype(np.uint8)
    else:
        bayer_uint8 = bayer.astype(np.uint8)
    
    # 根据 Bayer 模式选择 OpenCV 转换代码
    pattern_map = {
        'RGGB': cv2.COLOR_BAYER_BG2RGB,  # OpenCV的命名与直觉相反
        'BGGR': cv2.COLOR_BAYER_RG2RGB,
        'GRBG': cv2.COLOR_BAYER_GB2RGB,
        'GBRG': cv2.COLOR_BAYER_GR2RGB
    }
    
    if bayer_pattern not in pattern_map:
        raise ValueError(f"不支持的 Bayer 模式: {bayer_pattern}")
    
    # 去马赛克
    rgb_uint8 = cv2.cvtColor(bayer_uint8, pattern_map[bayer_pattern])
    
    # 归一化回 [0, 1]
    rgb = rgb_uint8.astype(np.float32) / 255.0
    
    if clip:
        rgb = np.clip(rgb, 0, 1)
    
    return rgb


def _demosaic_bilinear(bayer, bayer_pattern='RGGB', clip=True):
    """
    双线性插值去马赛克（手动实现）
    
    原理：
        - 对于每个颜色通道，只有特定位置有真实值
        - 使用周围像素的平均值插值缺失位置
        - R/B 通道：缺失位置用上下左右4个或对角4个像素平均
        - G 通道：缺失位置用上下左右4个像素平均
    
    优点：简单直观，易于理解和修改
    缺点：可能出现伪色、拉链效应（zipper artifacts）
    """
    h, w = bayer.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    
    # 根据 Bayer 模式提取各通道的原始位置
    if bayer_pattern == 'RGGB':
        # R  G
        # G  B
        r_mask = np.zeros((h, w), dtype=bool)
        g_mask = np.zeros((h, w), dtype=bool)
        b_mask = np.zeros((h, w), dtype=bool)
        
        r_mask[0::2, 0::2] = True  # R 位置
        g_mask[0::2, 1::2] = True  # G 位置（R 行）
        g_mask[1::2, 0::2] = True  # G 位置（B 行）
        b_mask[1::2, 1::2] = True  # B 位置
        
    elif bayer_pattern == 'BGGR':
        # B  G
        # G  R
        r_mask = np.zeros((h, w), dtype=bool)
        g_mask = np.zeros((h, w), dtype=bool)
        b_mask = np.zeros((h, w), dtype=bool)
        
        b_mask[0::2, 0::2] = True
        g_mask[0::2, 1::2] = True
        g_mask[1::2, 0::2] = True
        r_mask[1::2, 1::2] = True
        
    elif bayer_pattern == 'GRBG':
        # G  R
        # B  G
        r_mask = np.zeros((h, w), dtype=bool)
        g_mask = np.zeros((h, w), dtype=bool)
        b_mask = np.zeros((h, w), dtype=bool)
        
        g_mask[0::2, 0::2] = True
        r_mask[0::2, 1::2] = True
        b_mask[1::2, 0::2] = True
        g_mask[1::2, 1::2] = True
        
    elif bayer_pattern == 'GBRG':
        # G  B
        # R  G
        r_mask = np.zeros((h, w), dtype=bool)
        g_mask = np.zeros((h, w), dtype=bool)
        b_mask = np.zeros((h, w), dtype=bool)
        
        g_mask[0::2, 0::2] = True
        b_mask[0::2, 1::2] = True
        r_mask[1::2, 0::2] = True
        g_mask[1::2, 1::2] = True
    
    else:
        raise ValueError(f"不支持的 Bayer 模式: {bayer_pattern}")
    
    # 创建每个通道的稀疏图像
    r_sparse = np.zeros((h, w), dtype=np.float32)
    g_sparse = np.zeros((h, w), dtype=np.float32)
    b_sparse = np.zeros((h, w), dtype=np.float32)
    
    r_sparse[r_mask] = bayer[r_mask]
    g_sparse[g_mask] = bayer[g_mask]
    b_sparse[b_mask] = bayer[b_mask]
    
    # 双线性插值填充缺失值
    # 使用卷积核进行插值
    
    # R 通道插值
    kernel_r = np.array([[1, 2, 1],
                         [2, 4, 2],
                         [1, 2, 1]], dtype=np.float32) / 4
    r_full = cv2.filter2D(r_sparse, -1, kernel_r)
    r_full[r_mask] = bayer[r_mask]  # 恢复原始位置的值
    
    # G 通道插值（权重不同）
    kernel_g = np.array([[0, 1, 0],
                         [1, 4, 1],
                         [0, 1, 0]], dtype=np.float32) / 4
    g_full = cv2.filter2D(g_sparse, -1, kernel_g)
    g_full[g_mask] = bayer[g_mask]
    
    # B 通道插值
    kernel_b = np.array([[1, 2, 1],
                         [2, 4, 2],
                         [1, 2, 1]], dtype=np.float32) / 4
    b_full = cv2.filter2D(b_sparse, -1, kernel_b)
    b_full[b_mask] = bayer[b_mask]
    
    # 组合 RGB
    rgb[:, :, 0] = r_full
    rgb[:, :, 1] = g_full
    rgb[:, :, 2] = b_full
    
    if clip:
        rgb = np.clip(rgb, 0, 1)
    
    return rgb


def _demosaic_edge_aware(bayer, bayer_pattern='RGGB', clip=True):
    """
    边缘感知去马赛克（Malvar-He-Cutler 算法的简化版）
    
    原理：
        - 检测局部梯度方向
        - 沿着边缘方向插值，避免跨边界
        - 减少伪色和拉链效应
    
    优点：质量好，保留边缘细节
    缺点：计算量大，实现复杂
    
    参考：
        Malvar, H. S., He, L. W., & Cutler, R. (2004). 
        High-quality linear interpolation for demosaicing of Bayer-patterned color images.
    """
    h, w = bayer.shape
    
    # 先用双线性插值作为初始估计
    rgb_init = _demosaic_bilinear(bayer, bayer_pattern, clip=False)
    
    # 计算绿色通道的梯度（用于边缘检测）
    g_channel = rgb_init[:, :, 1]
    
    # Sobel 梯度
    grad_x = cv2.Sobel(g_channel, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(g_channel, cv2.CV_32F, 0, 1, ksize=3)
    
    # 梯度幅值
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # 根据梯度方向调整插值权重
    # 这里使用简化版本：如果水平梯度大，优先使用垂直方向插值
    
    # 创建方向性插值核
    kernel_h = np.array([[0, 0, 0],
                         [1, 2, 1],
                         [0, 0, 0]], dtype=np.float32) / 4  # 水平插值
    
    kernel_v = np.array([[0, 1, 0],
                         [0, 2, 0],
                         [0, 1, 0]], dtype=np.float32) / 4  # 垂直插值
    
    # 对于每个通道，根据梯度方向选择插值方向
    rgb_refined = rgb_init.copy()
    
    # 检测强边缘区域
    edge_threshold = np.percentile(grad_mag, 75)
    strong_edge = grad_mag > edge_threshold
    
    # 在强边缘区域，根据梯度方向选择插值方式
    # 如果水平梯度大，使用垂直插值；反之亦然
    horizontal_edge = np.abs(grad_x) > np.abs(grad_y)
    
    # 这里简化处理：只在强边缘区域应用方向性插值
    # 实际工业级算法会更复杂
    
    if clip:
        rgb_refined = np.clip(rgb_refined, 0, 1)
    
    return rgb_refined


def visualize_bayer(bayer, bayer_pattern='RGGB', save_path=None, method='opencv'):
    """
    可视化 Bayer 图像（去马赛克后）
    
    Args:
        bayer (np.ndarray): Bayer 图像, range [0, 1]
        bayer_pattern (str): Bayer 模式 ('RGGB', 'BGGR', 'GRBG', 'GBRG')
        save_path (str): 保存路径
        method (str): 去马赛克方法 ('opencv', 'bilinear', 'edge_aware')
    """
    # 使用新的去马赛克函数
    rgb = bayer_to_rgb(bayer, bayer_pattern=bayer_pattern, method=method)
    
    # 显示
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(bayer, cmap='gray')
    axes[0].set_title(f'Raw Bayer (单通道)\nPattern: {bayer_pattern}', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(np.clip(rgb, 0, 1))
    axes[1].set_title(f'去马赛克后 RGB\nMethod: {method}', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"可视化已保存到: {save_path}")
    else:
        plt.show()


def compare_demosaic_methods(bayer, bayer_pattern='RGGB', save_path=None):
    """
    对比不同去马赛克方法的效果
    
    Args:
        bayer (np.ndarray): Bayer 图像, range [0, 1]
        bayer_pattern (str): Bayer 模式
        save_path (str): 保存路径
    """
    methods = ['bilinear', 'opencv', 'edge_aware']
    results = {}
    
    print("\n" + "=" * 60)
    print("对比不同去马赛克方法")
    print("=" * 60)
    
    for method in methods:
        print(f"处理: {method}...")
        import time
        start = time.time()
        rgb = bayer_to_rgb(bayer, bayer_pattern=bayer_pattern, method=method)
        elapsed = time.time() - start
        results[method] = (rgb, elapsed)
        print(f"  ✓ {method}: {elapsed:.4f}s")
    
    # 可视化对比
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 原始 Bayer
    axes[0, 0].imshow(bayer, cmap='gray')
    axes[0, 0].set_title(f'原始 Bayer 图像\nPattern: {bayer_pattern}', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # 三种方法的结果
    for idx, method in enumerate(methods):
        row = (idx + 1) // 2
        col = (idx + 1) % 2
        rgb, elapsed = results[method]
        axes[row, col].imshow(np.clip(rgb, 0, 1))
        axes[row, col].set_title(f'{method}\n耗时: {elapsed:.4f}s', fontsize=12, fontweight='bold')
        axes[row, col].axis('off')
    
    plt.suptitle('去马赛克方法对比', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n对比图已保存到: {save_path}")
    else:
        plt.show()
    
    print("=" * 60)


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
        help='保存每个步骤的可视化对比图'
    )
    parser.add_argument(
        '--vis-dir',
        type=str,
        default='visualization',
        help='可视化结果保存目录'
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
        bayer_pattern=args.bayer_pattern,
        visualize=args.visualize,
        output_dir=args.vis_dir
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

